# -*- coding: utf-8 -*-

from torch import nn
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, in_channels, 3, 1),
                nn.InstanceNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, in_channels, 3, 1),
                nn.InstanceNorm2d(in_channels),
                )
    def forward(self, x):
        output = x + self.block(x)
        return output

class GeneratorResNet(nn.Module):
    def __init__(self, n_residual_blocks):
        super(GeneratorResNet, self).__init__()
        residual_blocks = []
        for _ in range(n_residual_blocks):
            residual_blocks += [ResidualBlock(256)]
        self.model = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(3, 64, 7),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),
                *residual_blocks,
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, 1, 1),    # 1 x 64 x 128 x 128
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 3, 7),
                nn.Tanh()
                )
    def forward(self, x):
        output = self.model(x)
        return output
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalized=True):
            block = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
            if normalized:
                block.append(nn.InstanceNorm2d(out_channels))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        
        self.model = nn.Sequential(
                *discriminator_block(3, 64, normalized=False),  # 1 x 64 x 64 x 64
                *discriminator_block(64, 128),   # 1 x 128 x 32 x 32
                *discriminator_block(128, 256),  # 1 x 256 x 16 x 16
                *discriminator_block(256, 512),  # 1 x 512 x 8 x 8
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(512, 1, 4, 1, 1),    # 1 x 1 x 8 x 8
                nn.Sigmoid()
                )
        
    def forward(self, x):
        return self.model(x)
        

