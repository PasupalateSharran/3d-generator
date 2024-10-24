import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_points=1024):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_points * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, point_cloud):
        point_cloud = point_cloud.view(point_cloud.size(0), -1)
        validity = self.fc(point_cloud)
        return validity
