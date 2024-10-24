import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=128, num_points=1024):
        super(Generator, self).__init__()
        self.num_points = num_points
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )

    def forward(self, z):
        point_cloud = self.fc(z)
        point_cloud = point_cloud.view(-1, self.num_points, 3)
        return point_cloud
