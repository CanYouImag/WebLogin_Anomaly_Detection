import torch
import torch.nn as nn


class AnomalyDetectionMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2,
                 hidden_dims: list = None, dropout: float = 0.1):
        super(AnomalyDetectionMLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = AnomalyDetectionMLP(input_dim=10, num_classes=15)
    dummy_input = torch.randn(32, 10)
    output = model(dummy_input)
    print("模型输出形状:", output.shape)
    print("输出 logits (各类别分数)", output[0])
