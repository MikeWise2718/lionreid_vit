import torch.nn as nn


class LionCNN(nn.Module):
    """Custom CNN for lion re-identification (following Matlala et al. 2024).

    Architecture:
        Conv2d(1,32,3) + GELU + MaxPool(2)
        Conv2d(32,64,3) + GELU + MaxPool(2)
        Conv2d(64,128,3) + GELU + MaxPool(2)
        Flatten -> FC -> GELU -> Dropout -> FC(num_classes)
    """

    def __init__(self, num_classes: int = 6, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        # After 3x MaxPool(2) on 384x384: 384/8 = 48, so 128 * 48 * 48
        fc_in = 128 * 48 * 48
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
