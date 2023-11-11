import pdb
import torch
import torch.nn as nn
import numpy as np
from torchvision import utils as vutils
from network.common_layers import conv2d, convTranspose2d


class SLE(nn.Module):
    def __init__(self, input_low_res_featureMaps: int, input_high_res_featureMaps: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv2d_1 = conv2d(in_channels=input_low_res_featureMaps, out_channels= 256, kernel_size=(4, 4), stride=1, padding=0, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.conv2d_2 = conv2d(in_channels= 256, out_channels=input_high_res_featureMaps, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x_low, x_high = inputs
        x = self.pool(x_low)
        x = self.conv2d_1(x)
        x = self.lrelu(x)
        x = self.conv2d_2(x)
        x = self.sigmoid(x)

        return x * x_high
    
    
class UpSample(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='nearest') #nearest neighbor
        self.conv2d = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.glu = nn.GLU(dim=1)

        self.filters = out_channels // 2


    def forward(self, inputs):
        x = self.upsample(inputs)
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.glu(x)

        return x

class Output(nn.Module):
    def __init__(self, feature_maps):
        super().__init__()
        self.conv2d = conv2d(in_channels=feature_maps, out_channels=3, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        x = self.conv2d(inputs)
        x = self.tanh(x)
        
        return x




class Generator(nn.Module):
    def __init__(self, output_resolution: int):
        super(Generator, self).__init__()
        assert output_resolution in [256, 512, 1024], "Resolution must be 256x256, 512x512 or 1024x1024"
        self.output_resolution = output_resolution
        
        self.filters = 1024
        i = 6

        self.conv2dtranspose = convTranspose2d(in_channels=256, out_channels=self.filters, kernel_size=(4, 4), stride=1, padding=0, bias=False)
        self.batchnorm = nn.BatchNorm2d(self.filters)
        self.glu = nn.GLU(dim=1)

        self.upsample_8 = UpSample(in_channels=self.filters//2, out_channels=self.filters//2)
        self.upsample_16 = UpSample(self.filters//4, self.filters//4)
        self.upsample_32 = UpSample(self.filters//8, self.filters//4)
        self.upsample_64 = UpSample(self.filters//8, self.filters//8)
        self.upsample_128 = UpSample(self.filters//16, self.filters//16)
        self.upsample_256 = UpSample(self.filters//32, self.filters//32)
        if self.output_resolution > 256:
            i += 1
            self.upsample_512 = UpSample(self.filters//64)
            if self.output_resolution > 512:
                i += 1
                self.upsample_1024 = UpSample(self.filters//128)

        self.sle_4_64 = SLE(self.filters//2, self.upsample_64.filters)
        self.sle_8_128 = SLE(self.upsample_8.filters, self.upsample_128.filters)
        self.sle_16_256 = SLE(self.upsample_16.filters, self.upsample_256.filters)
        if self.output_resolution > 256:
            self.sle_32_512 = SLE(self.upsample_32.filters, self.upsample_512.filters)


        self.tanh = nn.Tanh()
        self.conv2d = conv2d(in_channels=self.filters//(2**i), out_channels=3, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        

    def forward(self, inputs):
        x = self.conv2dtranspose(inputs)
        x = self.batchnorm(x)
        x_4 = self.glu(x)
        
        
        x_8 = self.upsample_8(x_4)
        x_16 = self.upsample_16(x_8)
        x_32 = self.upsample_32(x_16)
        
        x_64 = self.upsample_64(x_32)
        x_sle_64 = self.sle_4_64([x_4, x_64])

        x_128 = self.upsample_128(x_sle_64)
        x_sle_128 = self.sle_8_128([x_8, x_128])

        x_256 = self.upsample_256(x_sle_128)
        #x_sle_256
        x = self.sle_16_256([x_16, x_256])

        if self.output_resolution > 256:
            x_512 = self.upsample_512(x)
            #x_sle_512
            x = self.sle_32_512([x_32, x_512])
            
            if self.output_resolution > 512:
                #x_1024
                x = self.upsample_1024(x)

        x = self.conv2d(x)
        image = self.tanh(x)
        return image




if __name__ == '__main__':
    model = Generator(1024)
    pred = model(torch.normal(0, 1, size=(1, 256, 1, 1)))
    iteration = 100
    #vutils.save_image(model(torch.normal(0, 1, size=(25, 256, 1, 1))).add(1).mul(0.5), 'generated2/%d.jpg'%iteration, nrow=5)

    pdb.set_trace()