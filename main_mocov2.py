import argparse
import math
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data

from MoCov2.builder import MoCo
from MoCov2.SupLoss import MoCoSupConLoss
from data.data_generator import CohSourceDataset, create_dataloader


class ResNetEncoder(nn.Module):
    def __init__(self, arch='resnet18', input_channels=3, matrix_size=16, num_classes=128):
        super().__init__()
        import torchvision.models as models

        # load pre-trained ResNet (only structure)
        if arch == 'resnet18':
            base = models.resnet18(weights=None)
        elif arch == 'resnet34':
            base = models.resnet34(weights=None)
        elif arch == 'resnet50':
            base = models.resnet50(weights=None)
        else:
            raise ValueError(f"wrong structure: {arch}")

        if matrix_size <= 32:
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = base.maxpool

        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # get feature dim
        if arch == 'resnet50':
            feat_dim = 2048
        else:
            feat_dim = 512

        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def train(train_loader, model, criterion, optimizer, epoch, args):
    # metric
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()
    end = time.time()

    # training procedure
    for i, (view1, view2, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        view1 = view1.cuda(args.gpu, non_blocking=True)
        view2 = view2.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)

        logits, labels, all_labels = model(view1=view1, view2=view2, labels=labels)

        loss = criterion(logits, labels, all_labels)

        losses.update(loss.item(), view1.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def main():
    parser = argparse.ArgumentParser(description="Pytorch MoCo for unknown coherent signal sources estimation")

    # data
    parser.add_argument("--matrix_size", type=int, default=16, help="covariance matrix size M, usually equal to sensors")
    parser.add_argument("--input_channels", type=int, default=3, help="input channels, usually 3")
    parser.add_argument("--shuffle", type=bool, default=True, help="shuffle the training data")
    parser.add_argument("--pin_memory", type=bool, default=True, help="use pin memory")
    parser.add_argument("--num_samples_per_class", type=int, default=8000, help="num samples per class")
    parser.add_argument("--K_max", type=int, default=10, help="max number of coherent signals")
    parser.add_argument("--min_angle_sep", type=float, default=2.0, help="min angle separation between DOA")
    parser.add_argument("--snr_min", type=int, default=-15, help="minimum SNR in dB")
    parser.add_argument("--snr_max", type=int, default=15, help="maximum SNR in dB")
    parser.add_argument("--DOA_max", type=float, default=60.0, help="maximum DOA")
    parser.add_argument("--DOA_min", type=float, default=-60.0, help="minimum DOA")
    parser.add_argument("--snap_max", type=int, default=400, help="maximum snapshots")
    parser.add_argument("--snap_min", type=int, default=20, help="minimum snapshots")
    parser.add_argument("--mode", type=str, default="partial", help="coherent signal type")

    # arch
    parser.add_argument("-a", "--arch", default="resnet34",
                        choices=["resnet18", "resnet34", "resnet50"], help="model architecture choices")

    # train
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs to train")
    parser.add_argument("--start-epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument("-b", "--batch-size", default=256, type=int, help="initial batch size")
    parser.add_argument("-lr", "--learning-rate", default=0.001, type=float, help="initial learning rate", dest="lr",)
    parser.add_argument("--schedule", default=[120, 160], nargs="*", type=int, help="learning rate schedule (when to drop lr by 10x)")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum of SGD solver")
    parser.add_argument("--wd", default=1e-4, type=float, dest="weight_decay", help="weight decay (default: 1e-4)")
    parser.add_argument("-j", "--workers", default=16, type=int, help="number of data loading workers (default: 16)")
    parser.add_argument("-p", "--print-freq", default=10, type=int, help="print frequency, which can be seen at terminal (default: 10)")
    parser.add_argument("--resume", default="", type=str, help="path to latest checkpoint (default: none)")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use, usually cuda:0.")

    # MoCo
    parser.add_argument("--moco-dim", default=128, type=int, help="feature dimension (default: 128)")
    parser.add_argument("--moco-k", default=4096, type=int, help="queue size; number of negative keys (default: 4096)")
    parser.add_argument("--moco-m", default=0.999, type=float, help="moco momentum of updating key encoder (default: 0.9)")
    parser.add_argument("--moco-t", default=0.1, type=float, help="softmax temperature (default: 0.1)")
    parser.add_argument("--mlp", action="store_true", help="use mlp head")
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    torch.cuda.set_device(args.gpu)
    print(f"GPU: {args.gpu}")

    # create encoder
    print(f"=> create encoder '{args.arch}', input: {args.input_channels}×{args.matrix_size}×{args.matrix_size}")

    if args.arch == "resnet34":
        def base_encoder(num_classes):
            return ResNetEncoder(args.arch, args.input_channels, args.matrix_size, num_classes)
    else:
        raise ValueError(f"You have to choose between resnet18, resnet34 and resnet50.")

    # create model
    model = MoCo(
        base_encoder=base_encoder,
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        mlp=args.mlp,
    )
    model = model.cuda(args.gpu)

    print(f"queue size: {args.moco_k}, feature dim: {args.moco_dim}")

    # create loss function and optimizer
    criterion = MoCoSupConLoss().cuda(args.gpu)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.resume and os.path.isfile(args.resume):
        print(f"=> load checkpoint '{args.resume}'")
        ckpt = torch.load(args.resume, map_location=f"cuda:{args.gpu}")
        args.start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])

    cudnn.benchmark = True

    #  create dataset for training
    dataset = CohSourceDataset(
        num_samples_per_class=args.num_samples_per_class,
        M=args.matrix_size,
        K_max=args.K_max,
        doa_range=(args.DOA_min, args.DOA_max),
        coh_mode=args.mode,
        snapshot_range=(args.snap_min, args.snap_max),
        snr_range=(args.snr_min, args.snr_max),
        min_angle_sep=args.min_angle_sep,
        seed=None,
    )

    train_loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
    )

    # warm up queue
    print("Warming up queue...")
    model.eval()
    with torch.no_grad():
        for i, (view1, view2, labels) in enumerate(train_loader):
            view2 = view2.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
            k = model.encoder_k(view2)
            k = nn.functional.normalize(k, dim=1)
            model._dequeue_and_enqueue(k, labels)
            valid = (model.queue_labels != -1).sum().item()
            print(f"  Warmup batch {i}: queue {valid}/{args.moco_k}")
            if valid >= args.moco_k:
                break
    model.train()
    print("Queue warmup done!\n")

    # training procedure
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)
        print(f"\nEpoch {epoch}, LR: {lr:.6f}")

        train(train_loader, model, criterion, optimizer, epoch, args)

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }, f"moco_doa_epoch_{epoch + 1:04d}.pth.tar")


if __name__ == "__main__":
    main()