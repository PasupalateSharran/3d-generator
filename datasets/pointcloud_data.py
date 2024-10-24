import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from path import Path

from utils.point_operations import PointSampler, Normalize, RandRotation_z, RandomNoise, ToTensor
from utils.visualization import read_obj

def default_transforms():
    return transforms.Compose([
        PointSampler(1024),
        Normalize(),
        ToTensor()
    ])

class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, transform=default_transforms()):
        self.root_dir = root_dir
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []

        for model_id in sorted(os.listdir(root_dir)):
            model_dir = root_dir/Path(model_id)/"models"
            obj_file = model_dir/"model_normalized.obj"
            if os.path.exists(obj_file):
                sample = {'pcd_path': obj_file}
                self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_obj(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud}
