import torch
import torch.nn as nn

class FiveCharCaptchaCNN(nn.Module):
    def __init__(self, num_char_classes=51, label_length=5):
        super().__init__()
        self.num_char_classes = num_char_classes
        self.label_length = label_length

        # CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),      # -> 32 x 64 x 192
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),     # -> 32 x 64 x 192
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),           # -> 32 x 32 x 96

            nn.Conv2d(32, 64, kernel_size=3, padding=1),     # -> 64 x 32 x 96
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),     # -> 64 x 32 x 96
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),           # -> 64 x 16 x 48

            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # -> 128 x 16 x 48
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),           # -> 128 x 8 x 24

            nn.Conv2d(128, 256, kernel_size=3, padding=1),   # -> 256 x 8 x 24
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Force feature map into 5 horizontal slots
            nn.AdaptiveAvgPool2d((1, 5))                     # -> 256 x 1 x 5
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),                                    # -> 256*5
            nn.Linear(256 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, label_length * num_char_classes)  # -> 5*51
        )

    def forward(self, x):
        x = self.features(x)                                 # [B, 256, 1, 5]
        x = self.classifier(x)                               # [B, 255]
        x = x.view(-1, self.label_length, self.num_char_classes)
        return x                                             # [B, 5, 51]