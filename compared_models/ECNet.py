import torch
import torch.nn as nn


class ECNet(nn.Module):
    def __init__(self,
                 num_classes: int = 5,
                 in_features: int = 16,
                 hidden_units: list = [8, 8],
                 ):
        super(ECNet, self).__init__()

        self.nets = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_units[0]),
            nn.SELU(),
            nn.Linear(in_features=hidden_units[0], out_features=hidden_units[1]),
            nn.SELU(),
            nn.Linear(in_features=hidden_units[1], out_features=num_classes),
            # nn.Softmax(dim=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nets(x)
        return x


if __name__ == '__main__':
    net = ECNet()
    input = torch.randn(10)
    output = net(input)
    print(output)