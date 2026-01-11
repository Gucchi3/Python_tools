import torch 
import torch.nn as nn


class tiny_cnn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn5   = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6   = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)
        
        # input 32x32x3 ---->>> 4x4x256
        self.fc    = nn.Linear(4*4*256, 10)
        
    def forward(self, x):
        x = self.relu0(self.bn0(self.conv0(x)))
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        
        x = torch.flatten(x, 1)
        
        out = self.fc(x)
        
        return out
        
        


if __name__ == "__main__":
    from torchinfo import summary
    summary(tiny_cnn(), input_size=(1,3,32,32))
