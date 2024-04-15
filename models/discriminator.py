import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.nn.utils import spectral_norm

import models.basicblock as B



'''from https://github.com/researchmm/TTSR/blob/master/loss/discriminator.py'''
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)


class Discriminator(nn.Module):
    def __init__(self, Width=160,Height=160):
        super(Discriminator, self).__init__()
        self.conv1 = conv3x3(1, 32)
        self.LReLU1 = nn.LeakyReLU(0.2)
        self.conv2 = conv3x3(32, 32, 2)
        self.LReLU2 = nn.LeakyReLU(0.2)
        self.conv3 = conv3x3(32, 64)
        self.LReLU3 = nn.LeakyReLU(0.2)
        self.conv4 = conv3x3(64, 64, 2)
        self.LReLU4 = nn.LeakyReLU(0.2)
        self.conv5 = conv3x3(64, 128)
        self.LReLU5 = nn.LeakyReLU(0.2)
        self.conv6 = conv3x3(128, 128, 2)
        self.LReLU6 = nn.LeakyReLU(0.2)
        self.conv7 = conv3x3(128, 256)
        self.LReLU7 = nn.LeakyReLU(0.2)
        self.conv8 = conv3x3(256, 256, 2)
        self.LReLU8 = nn.LeakyReLU(0.2)
        self.conv9 = conv3x3(256, 512)
        self.LReLU9 = nn.LeakyReLU(0.2)
        self.conv10 = conv3x3(512, 512, 2)
        self.LReLU10 = nn.LeakyReLU(0.2)
        
        self.fc1 = nn.Linear(Width//32 * Height//32 * 512, 1024)
        self.LReLU11 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = self.LReLU1(self.conv1(x))
        x = self.LReLU2(self.conv2(x))
        x = self.LReLU3(self.conv3(x))
        x = self.LReLU4(self.conv4(x))
        x = self.LReLU5(self.conv5(x))
        x = self.LReLU6(self.conv6(x))
        x = self.LReLU7(self.conv7(x))
        x = self.LReLU8(self.conv8(x))
        x = self.LReLU9(self.conv9(x))
        x = self.LReLU10(self.conv10(x))
        x = x.view(x.size(0), -1)
        x = self.LReLU11(self.fc1(x))
        x = self.fc2(x)
        
        return x
class DiscriminatorTSRGAN(nn.Module):
    def __init__(self, Width=64,Height=64):
        super(DiscriminatorTSRGAN, self).__init__()
        self.conv1 =  nn.Conv2d(1, 64, kernel_size=4, 
                     stride=2)
        self.LReLU1 = nn.LeakyReLU()
        self.conv2 =  nn.Conv2d(64, 128,kernel_size=4, 
                     stride=2 )
        self.conv3 = nn.Conv2d(128, 256,kernel_size=4, 
                     stride=2 )
        self.swish1 = nn.SiLU()
        self.conv4 =  nn.Conv2d(256, 512,kernel_size=4, 
                     stride=2 )
        self.LReLU2 = nn.LeakyReLU()
        self.conv5 =  nn.Conv2d(512, 1024,kernel_size=4, 
                     stride=2 )
        self.swish2 = nn.SiLU()
        self.conv6 =  nn.Conv2d(1024, 512,kernel_size=3, 
                     stride=1 )
        self.LReLU3 = nn.LeakyReLU()
        self.conv7 =  nn.Conv2d(512,256,kernel_size=3, 
                     stride=1 )
        self.LReLU4 = nn.LeakyReLU()
        self.conv8 =  nn.Conv2d(256,64,kernel_size=3, 
                     stride=1 )
        self.LReLU5 = nn.LeakyReLU()
        self.conv9 =  nn.Conv2d(64,256,kernel_size=3, 
                     stride=1)
        self.swish3 = nn.SiLU()
        self.fc1 = nn.Linear(Width//76 * Height//76 * 256, 512)
        self.LReLU6 = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.LReLU1(self.conv1(x))
        x = self.conv2(x)

        x = self.swish1(self.conv3(x))
        x = self.LReLU2(self.conv4(x))
        x = self.swish2(self.conv5(x))
        x = self.LReLU3(self.conv6(x))
        x = self.LReLU4(self.conv7(x))
        x = self.LReLU5(self.conv8(x))
        x = self.swish3(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.LReLU6(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        
        return x

# --------------------------------------------
# VGG style Discriminator with 96x96 input
# --------------------------------------------
class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_nc=1, base_nc=32, ac_type='BL'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode='C')
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode='C'+ac_type)
        # 48, 64
        conv2 = B.conv(base_nc, base_nc*2, kernel_size=3, stride=1, mode='C'+ac_type)
        conv3 = B.conv(base_nc*2, base_nc*2, kernel_size=4, stride=2, mode='C'+ac_type)
        # 24, 128
        conv4 = B.conv(base_nc*2, base_nc*4, kernel_size=3, stride=1, mode='C'+ac_type)
        conv5 = B.conv(base_nc*4, base_nc*4, kernel_size=4, stride=2, mode='C'+ac_type)
        # 12, 256
        conv6 = B.conv(base_nc*4, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv7 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 6, 512
        conv8 = B.conv(base_nc*8, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv9 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4,
                                     conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# VGG style Discriminator with 128x128 input
# --------------------------------------------
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc=3, base_nc=64, ac_type='BL'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode='C')
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode='C'+ac_type)
        # 64, 64
        conv2 = B.conv(base_nc, base_nc*2, kernel_size=3, stride=1, mode='C'+ac_type)
        conv3 = B.conv(base_nc*2, base_nc*2, kernel_size=4, stride=2, mode='C'+ac_type)
        # 32, 128
        conv4 = B.conv(base_nc*2, base_nc*4, kernel_size=3, stride=1, mode='C'+ac_type)
        conv5 = B.conv(base_nc*4, base_nc*4, kernel_size=4, stride=2, mode='C'+ac_type)
        # 16, 256
        conv6 = B.conv(base_nc*4, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv7 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 8, 512
        conv8 = B.conv(base_nc*8, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv9 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 4, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4,
                                     conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 4 * 4, 100), 
                                        nn.LeakyReLU(0.2, True), 
                                        nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# VGG style Discriminator with 192x192 input
# --------------------------------------------
class Discriminator_VGG_192(nn.Module):
    def __init__(self, in_nc=3, base_nc=64, ac_type='BL'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode='C')
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode='C'+ac_type)
        # 96, 64
        conv2 = B.conv(base_nc, base_nc*2, kernel_size=3, stride=1, mode='C'+ac_type)
        conv3 = B.conv(base_nc*2, base_nc*2, kernel_size=4, stride=2, mode='C'+ac_type)
        # 48, 128
        conv4 = B.conv(base_nc*2, base_nc*4, kernel_size=3, stride=1, mode='C'+ac_type)
        conv5 = B.conv(base_nc*4, base_nc*4, kernel_size=4, stride=2, mode='C'+ac_type)
        # 24, 256
        conv6 = B.conv(base_nc*4, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv7 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 12, 512
        conv8 = B.conv(base_nc*8, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv9 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 6, 512
        conv10 = B.conv(base_nc*8, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv11 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5,
                                     conv6, conv7, conv8, conv9, conv10, conv11)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 3 * 3, 100),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# SN-VGG style Discriminator with 128x128 input
# --------------------------------------------
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x



if __name__ == '__main__':
    model = Discriminator_VGG_96(in_nc=1).cuda()
    x = torch.rand(16,1,64, 64).cuda()
    out = model(x)
    print (tuple(x.size()[1:]),x.size()[0])
    summary(model, tuple(x.size()[1:]),x.size()[0])  # For a single image


