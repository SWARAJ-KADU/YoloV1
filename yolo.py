import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

architecture_config = [
    #tuple (kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, bias=False, **kwargs)#the batchnorm will center the values. So a bias is useless in the previous layer
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YoloV1(nn.Module):
    def __init__(self, in_channel = 3, **kwargs):
        super(YoloV1, self).__init__()
        self.architecture = architecture_config
        self.in_channel = in_channel
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channel = self.in_channel

        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channel, x[1], kernel_size = x[0], stride = x[2], padding = x[3])]
                in_channel = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeate = x[2]
                for i in range(num_repeate):
                    layers += [CNNBlock(in_channel, conv1[1], kernel_size = conv1[0], stride = conv1[2], padding = conv1[3])]
                    layers += [CNNBlock(conv1[1], conv2[1], kernel_size = conv2[0], stride = conv2[2], padding = conv2[3])]
                    in_channel = conv2[1]        

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S*S*(C + B*5))
        )
                
    