import torch
import torch.nn as nn


class ERNet(nn.Module):
    def __init__(self,
                 num_classes: int = 5,
                 in_features: int = 16,
                 hidden_units: list = [8, 8],
                 ):
        super(ERNet, self).__init__()

        self.nets = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_units[0]),
            nn.SELU(),
            nn.Linear(in_features=hidden_units[0], out_features=hidden_units[1]),
            nn.SELU(),
            nn.Linear(in_features=hidden_units[1], out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nets(x)
        return x


if __name__ == '__main__':
    net = ERNet()
    input = torch.randn(10)
    output = net(input)
    print(output)