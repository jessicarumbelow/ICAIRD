import numpy as np
# np.set_printoptions(precision=3, linewidth=150)
import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
import math
import torchvision.models as tv_models
import sys
from PIL import Image


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def save_image(arr, name):
    im = Image.fromarray(np.uint8(arr * 255))
    im.save('imgs/{}.jpg'.format(name))


class VGG16(nn.Module):
    def __init__(self, num_outputs, im_dim=(3, 256, 256), depth=None, widen_factor=None, dropout_rate=None, layer_channel=None):
        super().__init__()
        self.im_dim = im_dim
        self.model = tv_models.vgg16()
        self.model.features[0] = nn.Conv2d(im_dim[0], 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.model.classifier[6] = nn.Linear(4096, num_outputs)

    def forward(self, x):
        return self.model(x)


class WRN50_2(nn.Module):
    def __init__(self, num_outputs, im_dim=(3, 256, 256), depth=None, widen_factor=None, dropout_rate=None, layer_channel=None):
        super().__init__()
        self.im_dim = im_dim
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'wide_resnet50_2')

    def forward(self, x):
        return self.model(x)

class WideConv(nn.Module):

    def __init__(self, num_outputs, im_dim=(3, 256, 256), depth=None, widen_factor=None, dropout_rate=None, layer_channel=None):
        super().__init__()
        self.im_dim = im_dim
        self.features = nn.Sequential(nn.Conv2d(im_dim[0], 384, kernel_size=3),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(384, 384, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(384 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, num_outputs))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x[0]


class AlexNet(nn.Module):

    # AlexNet
    def __init__(self, num_outputs, im_dim=(1, 256, 256), depth=None, widen_factor=None, dropout_rate=None, layer_channel=None):
        super().__init__()
        self.im_dim = im_dim
        self.features = nn.Sequential(nn.Conv2d(im_dim[0], 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, num_outputs))

    def forward(self, x):

        x = x.transpose(1, -1)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x[0]


class FullyConnected(nn.Module):

    def __init__(self, num_outputs, num_hidden_layers=2, num_nodes=5, activation='nn.ReLU', output_activation='nn.Sigmoid', im_dim=(1, 256, 256)):
        super().__init__()
        self.activation = activation
        self.output_activation = output_activation
        self.im_dim = im_dim
        self.num_inputs = np.prod(np.array(im_dim))
        self.num_outputs = num_outputs
        self.num_nodes = num_nodes
        self.num_hidden_layers = num_hidden_layers
        self.network = self._get_name()

        self.build_network()

    def forward(self, x):
        return self.network(x)

    def build_network(self):
        layers = [nn.Linear(self.num_inputs, self.num_nodes), eval(self.activation)()] + ([nn.Linear(self.num_nodes, self.num_nodes), eval(self.activation)()] * (self.num_hidden_layers - 1)) + [
            nn.Linear(self.num_nodes, self.num_outputs), eval(self.output_activation)()]

        self.network = nn.Sequential(*layers)


class Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.bn(x)
        out = self.conv1(x)
        out = F.relu(out)
        out = torch.add(out, x)
        out = self.conv2(out)
        return out


class JessNet(nn.Module):

    def __init__(self, num_outputs, depth, widen_factor, dropout_rate, im_dim, layer_channel):
        super().__init__()
        self.im_dim = im_dim
        self.dropout = dropout_rate
        network = [nn.Conv2d(im_dim[0], int(layer_channel), kernel_size=3, padding=1)]
        for d in range(1, depth):
            network.append(nn.ReLU())
            network.append(nn.Conv2d(int(layer_channel), int(layer_channel * widen_factor), kernel_size=3, padding=1))
            layer_channel *= widen_factor
        self.network = nn.Sequential(*network)
        self.fc = nn.Linear(int(self.im_dim[1] * self.im_dim[2] * layer_channel), num_outputs)

    def forward(self, x):
        x = x.transpose(1, -1)
        num_samples = x.shape[0]
        out = self.network(x)
        out = F.dropout(out, self.dropout, self.training)
        out = out.view(num_samples, -1)
        out = self.fc(out)
        return out


# WideResNet implementation below based on https://github.com/murari023/WideResNet-pytorch/blob/master/wideresnet.py

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout_rate = dropout_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):

    def __init__(self, num_layers, in_planes, out_planes, block, stride, dropout_rate=0.0):
        super().__init__()
        self.layer = self.make_layer(block, in_planes, out_planes, num_layers, stride, dropout_rate)

    def make_layer(self, block, in_planes, out_planes, num_layers, stride, dropout_rate):
        layers = []
        for i in range(num_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, num_outputs, depth, widen_factor, dropout_rate, im_dim, layer_channel=16):
        super().__init__()
        self.im_dim = im_dim
        self.pooling_dim = 8
        self.num_outputs = num_outputs
        self.num_channels = [layer_channel, int(layer_channel * widen_factor), 2 * int(layer_channel * widen_factor), 4 * int(layer_channel * widen_factor)]
        n = depth
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(self.im_dim[0], self.num_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, self.num_channels[0], self.num_channels[1], block, 1, dropout_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, self.num_channels[1], self.num_channels[2], block, 2, dropout_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, self.num_channels[2], self.num_channels[3], block, 2, dropout_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(self.num_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(self.num_channels[3], num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, self.pooling_dim)
        out = out.view(-1, self.num_channels[3])
        out = self.fc(out)
        return out

class MNISTNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
