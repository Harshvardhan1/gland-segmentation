import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import torch.tensor
from torchvision import datasets, transforms
from torch.autograd import Variable

# Add your own dataset as data_loader

# Hyperparameters


def conv2x2(in_c, out, kernel_size=3, stride=1, padding=0, bias=True, useBN=True):
    if useBN:
        return nn.Sequential(
                  nn.Conv2d(in_c, out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                  nn.BatchNorm2d(out),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(out, out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                  nn.BatchNorm2d(out),
                  nn.LeakyReLU(0.2))

    else:
        return nn.Sequential(
                    nn.Conv2d(in_c, out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.ReLU(),
                    nn.Conv2d(out, out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    nn.ReLU()
                    )


def upsample(c, f):
    return nn.Sequential(
        nn.ConvTranspose2d(c, f, 2, 2, 0, bias=False),
        nn.ReLU())

def crop_cat(inp1,inp2):
    #import pdb;pdb.set_trace()
    offset = inp1.size()[2] - inp2.size()[2]
    padding = 2 * [offset // 2, offset // 2]
    out2 = F.pad(inp2, padding)
    return torch.cat([out2, inp1], 1)


class UNet(nn.Module):
    def __init__(self, useBN=True,in_channels=3,n_classes=2):
        super(UNet, self).__init__()

        self.conv1   = conv2x2(in_channels, 32, useBN=useBN)
        self.conv2   = conv2x2(32, 64, useBN=useBN)
        self.conv3   = conv2x2(64, 128, useBN=useBN)
        self.conv4   = conv2x2(128, 256, useBN=useBN)
        self.conv5   = conv2x2(256, 512, useBN=useBN)

        self.conv4m = conv2x2(512, 256, useBN=useBN)
        self.conv3m = conv2x2(256, 128, useBN=useBN)
        self.conv2m = conv2x2(128, 64, useBN=useBN)
        self.conv1m = conv2x2(64, 32, useBN=useBN)

        self.conv0  = nn.Conv2d(32, 1, 1, 1, 0)

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

        #self.dropout = nn.Dropout2d()

        ## weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                #torch.nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        output1 = self.max_pool(self.conv1(x))
        output2 = self.max_pool(self.conv2(output1))
        output3 = self.max_pool(self.conv3(output2))
        output4 = self.max_pool(self.conv4(output3))
        output5 = self.max_pool(self.conv5(output4))

        #import pdb;pdb.set_trace()
        #output5 = self.dropout(output5)

        conv5m_out = crop_cat(self.upsample54(output5), output4)
        conv4m_out = self.conv4m(conv5m_out)

        conv4m_out = crop_cat(self.upsample43(output4), output3)
        conv3m_out = self.conv3m(conv4m_out)

        conv3m_out = crop_cat(self.upsample32(output3), output2)
        conv2m_out = self.conv2m(conv3m_out)

        conv2m_out = crop_cat(self.upsample21(output2), output1)
        conv1m_out = self.conv1m(conv2m_out)

        final = self.conv0(conv1m_out)
        return final