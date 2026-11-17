"""Five CIFAR-10 models (3 x 32 x 32 -> 10 classes) from ~6k to ~100k parameters.

Selected with ``--model`` (``MODEL=`` in run_lightsecagg_distributed_pytorch.sh); d = number of parameters:

    name      layers                                                                     parameters
    cnn6k     3 x [conv3x3, ReLU, maxpool] (12, 16, 24) + global avg pool + FC                5,810
    cnn15k    3 x [conv3x3, ReLU, maxpool] (16, 32, 32) + global avg pool + FC               14,666
    lr        logistic regression: one linear layer on the 3072 pixels                      30,730
    lenet5    LeNet-5: conv5x5 (6), maxpool, conv5x5 (16), maxpool, FC 120, FC 84, FC        62,006
    mlp99k    2-layer perceptron: FC 3072 -> 32, ReLU, FC 32 -> 10                          98,666

(a linear layer with n_in inputs and n_out outputs has n_in * n_out weights + n_out biases;
a convolution only k * k * c_in * c_out + c_out, independent of the image size.)

No batch norm on purpose: its running statistics and the integer counter num_batches_tracked
are buffers of the state_dict, i.e. they would be quantized, masked and aggregated like weights.
Every entry of the state_dict here is a trainable parameter.
"""
import torch.nn as nn

from fedml_api.model.linear.lr import LogisticRegression

IMAGE_SHAPE = (3, 32, 32)
PIXELS = 3 * 32 * 32


class ConvGAPNet(nn.Module):
    """[conv3x3, ReLU, maxpool 2] per channel width, global average pooling, linear classifier"""

    def __init__(self, channels, num_classes=10, in_channels=3):
        super().__init__()
        layers, c = [], in_channels
        for c_out in channels:
            layers += [nn.Conv2d(c, c_out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            c = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


class LeNet5(nn.Module):
    """LeNet-5 (LeCun et al. 1998) for 3 x 32 x 32 inputs, with ReLU"""

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels, 6, 5), nn.ReLU(), nn.MaxPool2d(2),   # 6 x 14 x 14
                                      nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2))            # 16 x 5 x 5
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
                                        nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


CIFAR10_MODELS = {
    "cnn6k": lambda num_classes=10: ConvGAPNet((12, 16, 24), num_classes),
    "cnn15k": lambda num_classes=10: ConvGAPNet((16, 32, 32), num_classes),
    "lr": lambda num_classes=10: nn.Sequential(nn.Flatten(), LogisticRegression(PIXELS, num_classes)),
    "lenet5": lambda num_classes=10: LeNet5(num_classes),
    "mlp99k": lambda num_classes=10: nn.Sequential(nn.Flatten(), nn.Linear(PIXELS, 32), nn.ReLU(),
                                                   nn.Linear(32, num_classes)),
}
