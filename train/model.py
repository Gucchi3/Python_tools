import torch
import torch.nn as nn
from torchinfo import summary

# class TinyCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # --- Layer 1 & 2 ---
#         # Conv: 3ch -> 32ch, 5x5 kernel, stride 1, padding 2 (to keep 32x32)
#         # MaxPool: 32x32 -> 16x16
#         self.conv0 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
#         self.bn0   = nn.BatchNorm2d(32)
#         self.relu0 = nn.ReLU(inplace=True)
#         self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # --- Layer 3 & 4 ---
#         # Conv: 32ch -> 32ch, 5x5 kernel, stride 1, padding 2 (to keep 16x16)
#         # MaxPool: 16x16 -> 8x8
#         self.conv1 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
#         self.bn1   = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # --- Layer 5 & 6 ---
#         # Conv: 32ch -> 64ch, 5x5 kernel, stride 1, padding 2 (to keep 8x8)
#         # MaxPool: 8x8 -> 4x4
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
#         self.bn2   = nn.BatchNorm2d(64)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # --- Layer 7 (Fully Connected) ---
#         # Input: 4x4 x 64ch = 1024
#         # Output: 10 classes (CIFAR-10)
#         self.fc = nn.Linear(64 * 4 * 4, 10)
        
#     def forward(self, x):
#         # Block 1
#         x = self.conv0(x)
#         x = self.bn0(x)
#         x = self.relu0(x)
#         x = self.pool0(x)
        
#         # Block 2
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
        
#         # Block 3
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
        
#         # Flatten
#         x = torch.flatten(x, 1)
        
#         # Classifier
#         out = self.fc(x)
        
#         return out



class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- Layer 1 & 2 ---
        # Conv: 3ch -> 32ch, 5x5 kernel, stride 1, padding 2 (to keep 32x32)
        # MaxPool: 32x32 -> 16x16
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn0   = nn.BatchNorm2d(32)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # --- Layer 3 & 4 ---
        # Conv: 32ch -> 32ch, 5x5 kernel, stride 1, padding 2 (to keep 16x16)
        # MaxPool: 16x16 -> 8x8
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # --- Layer 5 & 6 ---
        # Conv: 32ch -> 64ch, 5x5 kernel, stride 1, padding 2 (to keep 8x8)
        # MaxPool: 8x8 -> 4x4
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # --- Layer 7 (Fully Connected) ---
        # Input: 4x4 x 64ch = 1024
        # Output: 10 classes (CIFAR-10)
        self.fc = nn.Linear(64 * 4 * 4, 10)
        
    def forward(self, x):
        # Block 1
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        #x = self.pool0(x)
        
        # Block 2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        #x = self.pool1(x)
        
        # Block 3
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        #x = self.pool2(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classifier
        out = self.fc(x)
        
        return out



    
    
if __name__ == "__main__":
    model = TinyCNN()
    
    # 構造の確認
    summary(model, input_size=(1, 3, 32, 32))