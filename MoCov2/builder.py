import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
            self,
            base_encoder,
            dim: int = 128,
            K: int = 4096,
            m: float = 0.999,
            T: float = 0.1,
            mlp: bool = True,
    ) -> None:
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 4096)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.1)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = base_encoder(num_classes=dim)  # query encoder
        self.encoder_k = base_encoder(num_classes=dim)  # key encoder

        if mlp:  # MoCo v2: add MLP projection head
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        # Copy the parameters of the query encoder to initialize the key encoder
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # key encoder need no gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))  # shape: [dim, K]
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_labels", torch.full((K,), -1, dtype=torch.long))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder
        """

        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels) -> None:
        """
        Update queue: Enqueue new keys and dequeue old ones
        """

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # Deal with the situation where batch_size cannot be divided by K
        if ptr + batch_size > self.K:
            # It needs to be written in two segments (circular queue)
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue_labels[ptr:] = labels[:remaining]
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            self.queue_labels[:batch_size - remaining] = labels[remaining:]
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.queue_labels[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr


    def forward(self, view1, view2, labels):
        """
        view1: (B, 3, M, M) query
        view2: (B, 3, M, M) key
        labels: (B,) for supervised learning
        """

        # compute query features
        q = self.encoder_q(view1)  # compute Query's feature   shape: (B, dim)
        q = nn.functional.normalize(q, dim=1)  # L2 norm

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(view2)  # (B, dim)
            k = nn.functional.normalize(k, dim=1)  # L2 norm

        all_keys = torch.cat([k, self.queue.clone().detach().T], dim=0)  # (dim, B+Q)
        all_labels = torch.cat([labels, self.queue_labels.clone().detach()])  # (B+Q,)

        # # compute logits
        logits = torch.einsum("bd, nd -> bn ", q, all_keys)  # (B, dim) @ (dim, B+Q) = (B, B+Q)

        # apply temperature
        logits /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, labels)

        return logits, labels, all_labels