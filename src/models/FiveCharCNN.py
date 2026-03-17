import torch
import torch.nn as nn
import torch.nn.functional as F

class FiveCharCaptchaCNN(nn.Module):
    def __init__(self, num_char_classes=51, label_length=5):
        super(FiveCharCaptchaCNN, self).__init__()
        self.num_char_classes = num_char_classes
        self.label_length = label_length

        #Convolutional architecture
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),   # Output: 16 x 64 x 192
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1),   # Output: 32 x 64 x 192
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),          # Output: 32 x 32 x 192

            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),   # Output: 64 x 32 x 192
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),                  # Output: 64 x 16 x 96

            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),  # Output: 128 x 16 x 96
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),                  # Output: 128 x 8 x 48

            nn.Conv2d(128,128, kernel_size=3, stride=2, padding=1), # Output: 128 x 4 x 24
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))           # Output: 128 x 4 x 8
        )

        #Classifier architecture
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the output from convolutional layers
            nn.Linear(128*4*8, 512),
            nn.ReLU(),
            nn.Dropout(0.38),

            nn.Linear(512, num_char_classes * self.label_length)  # Output: 51 * 5 = 255
        )
    def forward(self, x):
        x = self.convolution(x)
        x = self.classifier(x)
        return x.view(-1, self.label_length, self.num_char_classes)  # Reshape to (batch_size, label_length, num_char_classes)