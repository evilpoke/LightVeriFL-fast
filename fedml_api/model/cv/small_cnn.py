"""Small CNNs for CIFAR-10 (3 x 32 x 32 -> 10 classes) with at most 10,000 parameters.

Named by their (rounded) number of parameters, selected with ``--model`` (``MODEL=`` in
run_lightsecagg_distributed_pytorch.sh):

    name      layers                                                         parameters
    cnn1k     2 x [conv3x3, ReLU, maxpool] (8, 12)      + global avg pool + FC    1,230
    cnn4k     3 x [conv3x3, ReLU, maxpool] (8, 16, 16)  + global avg pool + FC    3,882
    cnn6k     3 x [conv3x3, ReLU, maxpool] (12, 16, 24) + global avg pool + FC    5,810
    lenet8k   LeNet-5 style: 2 x [conv5x5, ReLU, maxpool] (6, 8) + FC 32 + FC     8,426
    cnn10k    4 x [conv3x3, ReLU, maxpool] (8, 16, 20, 28) + global avg pool + FC 9,650

No batch norm on purpose: its running statistics and the integer counter num_batches_tracked
are buffers of the state_dict, i.e. they would be quantized, masked and aggregated like weights.
Every entry of the state_dict here is a trainable parameter (d = number of parameters).
"""
import torch.nn as nn


def _conv_block(c_in, c_out, k=3):
    return [nn.Conv2d(c_in, c_out, k, padding=k // 2), nn.ReLU(), nn.MaxPool2d(2)]


class ConvGAPNet(nn.Module):
    """[conv3x3, ReLU, maxpool 2] per channel width, global average pooling, linear classifier"""

    def __init__(self, channels, num_classes=10, in_channels=3):
        super().__init__()
        layers, c = [], in_channels
        for c_out in channels:
            layers += _conv_block(c, c_out)
            c = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


class LeNetSmall(nn.Module):
    """LeNet-5 layout (conv5x5 -> pool -> conv5x5 -> pool -> FC -> FC) with narrow layers"""

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels, 6, 5), nn.ReLU(), nn.MaxPool2d(2),   # 6 x 14 x 14
                                      nn.Conv2d(6, 8, 5), nn.ReLU(), nn.MaxPool2d(2))             # 8 x 5 x 5
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(8 * 5 * 5, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


SMALL_CIFAR10_CNNS = {
    "cnn1k": lambda num_classes=10: ConvGAPNet((8, 12), num_classes),
    "cnn4k": lambda num_classes=10: ConvGAPNet((8, 16, 16), num_classes),
    "cnn6k": lambda num_classes=10: ConvGAPNet((12, 16, 24), num_classes),
    "lenet8k": lambda num_classes=10: LeNetSmall(num_classes),
    "cnn10k": lambda num_classes=10: ConvGAPNet((8, 16, 20, 28), num_classes),
}
