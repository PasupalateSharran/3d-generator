# 3D Bench Mesh Generator

This project implements a 3D GAN (Generative Adversarial Network) to generate point clouds and mesh models of benches for use in game assets. The model is trained on point clouds extracted from OBJ files, and the generator creates new 3D bench models from latent vectors (random noise).

## Project Structure

```bash
bench_generator/
│
├── data/
│   └── Bench/                       # Dataset: 3D models of benches in OBJ format
│
├── models/
│   ├── generator.py                 # Generator model (3D GAN)
│   └── discriminator.py             # Discriminator model (3D GAN)
│
├── datasets/
│   └── pointcloud_data.py           # Dataset and data loader for 3D point clouds
│
├── training/
│   └── train.py                     # Training script for GAN
│
├── utils/
│   ├── point_operations.py          # Utilities for point cloud processing
│   └── visualization.py             # Visualization functions for meshes and point clouds
│
├── checkpoints/                     # Model checkpoints saved during training
│   └── generator_checkpoint.pth
│
├── scripts/
│   └── generate_bench_mesh.py       # Script to generate and save a new bench mesh
│
├── requirements.txt                 # Required Python libraries
└── README.md                        # Project documentation
```

## Dataset

You will need a dataset of 3D benches in the .obj format. The dataset should be placed in the data/Bench/ directory. Each bench should be stored as an individual folder with its OBJ file.

## Training the model

```bash
cd training
python train.py
```

This will start the training process, using the dataset of benches and generating new 3D models of benches over time. The script saves checkpoints of the generator model in the checkpoints/ folder every few epochs.

### Key Hyperparameters

You can adjust key hyperparameters like the batch size, learning rate, and latent dimension inside the train.py script. Currently, it defaults to:

Latent dimension: 128
Batch size: 32
Learning rate: 0.0002

## Generating a 3D Mesh

After training, you can generate a new 3D bench model by running the generate_bench_mesh.py script:

```bash
cd scripts
python generate_bench_mesh.py
```

This script will use the trained generator model (loaded from the checkpoints/ folder) to generate a new point cloud. The generated point cloud will then be converted into a 3D mesh and saved as an .obj file.

The generated file will be saved as generated_bench.obj, and the mesh will be displayed using Open3D.

## Visualization

To visualize point clouds and meshes, the visualization.py file in the utils/ folder provides utility functions to:

Load and display OBJ files.
Visualize point clouds in 3D.
Rotate and inspect generated point clouds.
You can call these functions in your custom scripts or while debugging.

## File Explanations

generator.py: Defines the architecture of the 3D GAN generator, which creates 3D models from latent vectors.

discriminator.py: Defines the architecture of the 3D GAN discriminator, which distinguishes between real and generated models.

pointcloud_data.py: Implements the data loader for handling 3D point clouds and OBJ files.

train.py: Main script for training the 3D GAN model.
generate_bench_mesh.py: Script to generate a new bench model using the trained generator.

point_operations.py: Includes utility functions for normalizing point clouds, adding noise, and other point operations.

visualization.py: Provides functions to visualize the point clouds and meshes in 3D.

## Future Work

Add more diverse classes of furniture to generate different types of models (e.g., chairs, tables).
Experiment with higher-resolution models.
Improve the mesh post-processing step for better model quality.
