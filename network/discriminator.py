import pdb
import torch
import torch.nn as nn
from network._aux import center_crop_image
from network.common_layers import conv2d


class Input(nn.Module):
    def __init__(self, downsample_factor: int, filters: int):
        super().__init__()
        assert downsample_factor in [1, 2, 4]

        conv1s, conv2s = 2, 2
        p1=1

        if downsample_factor <= 2:
            conv2s = 1
            p1 = 2
        if downsample_factor == 1:
            conv1s = 1
        self.filters = filters

        self.conv2d_1 = conv2d(in_channels=3, out_channels=self.filters, kernel_size=(4, 4), stride=conv1s, padding=p1, bias=False)
        self.conv2d_2 = conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=(4, 4), stride=conv2s, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(self.filters)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    
    def forward(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.lrelu(x)
        x = self.conv2d_2(x)
        x = self.batchnorm(x)
        x = self.lrelu(x)

        return x
    
class DownSample1(nn.Module):
    def __init__(self, c_in:int, c_out:int):
        super().__init__()

        self.conv2d_1 = conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(4, 4), stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(c_out)
        self.conv2d_2 = conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(c_out)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.batchnorm1(x)
        x = self.lrelu(x)
        x = self.conv2d_2(x)
        x = self.batchnorm2(x)
        x = self.lrelu(x)

        return x
    

class DownSample2(nn.Module):
    def __init__(self, c_in:int, c_out:int):
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv2d_1 = conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.batchnorm = nn.BatchNorm2d(c_out)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    
    def forward(self, inputs):
        x = self.pool(inputs)
        x = self.conv2d_1(x)
        x = self.batchnorm(x)
        x = self.lrelu(x)

        return x
    

class DownSample(nn.Module):
    def __init__(self, c_in:int, c_out:int):
        super().__init__()

        self.downsample1 = DownSample1(c_in, c_out)
        self.downsample2 = DownSample2(c_in, c_out)

    def forward(self, inputs):
        x1 = self.downsample1(inputs)
        x2 = self.downsample2(inputs)

        return x1 + x2
    

class DecoderBlock(nn.Module):
    def __init__(self, feature_maps:int):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='nearest')
        self.conv2d = conv2d(in_channels=feature_maps, out_channels=feature_maps, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(feature_maps)
        self.glu = nn.GLU(dim=1)

    def forward(self, inputs):
        x = self.upsample(inputs)
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.glu(x)

        return x
    
class SimpleDecoder(nn.Module):
    def __init__(self, feature_maps):
        super().__init__()

        self.decoder_block_filter_sizes = [feature_maps, feature_maps//2, feature_maps//4, feature_maps//8]
        self.decoder_blocks = nn.ModuleList([DecoderBlock(feature_maps=x) for x in self.decoder_block_filter_sizes])

        self.conv2d = conv2d(in_channels=feature_maps//16, out_channels=3, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()


    def forward(self, inputs):
        x = inputs
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        x = self.conv2d(x)
        x = self.tanh(x)

        return x
    

class Output(nn.Module):
    def __init__(self, feature_maps: int):
        super().__init__()

        self.conv2d_1 = conv2d(in_channels=feature_maps, out_channels=feature_maps, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.batchnorm = nn.BatchNorm2d(feature_maps)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2d_2 = conv2d(in_channels=feature_maps, out_channels=1, kernel_size=(4, 4), stride=1, padding=0, bias=False)


    def forward(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.batchnorm(x)
        x = self.lrelu(x)
        x = self.conv2d_2(x)

        return x



class Discriminator(nn.Module):
    def __init__(self, input_resolution: int):
        super(Discriminator, self).__init__()

        assert input_resolution in [256, 512, 1024], "Resolution must be 256x256, 512x512 or 1024x1024"

        self.filters = 8

        self.input = Input(input_resolution/256, filters=self.filters)

        self.downsample_128 = DownSample(c_in=self.filters, c_out=64)
        self.downsample_64 = DownSample(c_in=64, c_out=128)
        self.downsample_32 = DownSample(c_in=128 , c_out=128)
        self.downsample_16 = DownSample(c_in=128 , c_out=256)
        self.downsample_8 = DownSample(c_in=256 , c_out=512)


        self.decoder_image_part = SimpleDecoder(256)
        self.decoder_image = SimpleDecoder(512)

        self.output = Output(feature_maps=512)

    def forward(self, inputs, label):
        x = self.input(inputs)

        x = self.downsample_128(x)
        x = self.downsample_64(x)
        x = self.downsample_32(x)
        x_16 = self.downsample_16(x)
        x_8 = self.downsample_8(x_16)

        x_real_fake_logits = self.output(x_8)

        if label=='real':
            x_16_crop = center_crop_image(x_16, 8)
            x_16_crop_decoded_128 = self.decoder_image_part(x_16_crop)
            x_8_decoded_128 = self.decoder_image(x_8)

            return x_real_fake_logits, x_8_decoded_128, x_16_crop_decoded_128
        
        return x_real_fake_logits
        
    



if __name__ == '__main__':
    model = Discriminator(1024)
    pred = model(torch.normal(0, 1, size=(1, 3, 1024, 1024)), label='real')
    pdb.set_trace()

