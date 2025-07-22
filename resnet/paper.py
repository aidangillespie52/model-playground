import torch.nn as nn
import torch

# 34 layer residual
class Block(nn.Module):
    def __init__(self, downsample=False, in_channels=None):
        super().__init__()
        self.downsample = downsample
        first_stride = 2 if downsample else 1
        
        out_channels = in_channels*first_stride
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=first_stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        og = x
        
        if self.downsample:
            og = self.downsample_conv(og)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
    
        x = x + og
        x = self.relu(x)
        
        return x

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        channels = 64
        self.l0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    
        )
        
        self.l1, channels = self.create_layer(num_blocks=3, channels=channels)
        self.l2, channels = self.create_layer(num_blocks=4, channels=channels, downsample=True)
        self.l3, channels = self.create_layer(num_blocks=6, channels=channels, downsample=True)
        self.l4, channels = self.create_layer(num_blocks=3, channels=channels, downsample=True)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # flattens to the number of channels
        self.fc1 = nn.Linear(channels, 1000)
        self.fc2 = nn.Linear(1000, 10)
        
        self.sm = nn.Softmax(dim=1) # put in dim for depreciation warning
        self.relu = nn.ReLU(inplace=False)
    
    # kind of a weird way to do it but it works and isn't super importanbt
    def create_layer(self, num_blocks, channels=None, downsample=False):
        layers = []
        
        init_block = Block(downsample=downsample, in_channels=channels)
        layers.append(init_block)
        
        if downsample:
            channels *= 2
        
        for num_blocks in range(num_blocks-1):
            layers.append(Block(in_channels=channels))
        
        return nn.Sequential(
            *layers
        ), channels
        
    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        #x = self.sm(x)
        return x
        
if __name__ == '__main__':
    rn = ResNet()
    r = torch.rand(1, 3, 448, 448)
    print(rn(r))