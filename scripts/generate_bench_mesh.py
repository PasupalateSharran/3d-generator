import torch
import numpy as np
import open3d as o3d
from models.generator import Generator

def generate_bench_mesh(generator, latent_dim, device):
    generator.eval()
    z = torch.randn((1, latent_dim)).to(device)

    with torch.no_grad():
        generated_point_cloud = generator(z).cpu().numpy()
    generated_point_cloud = generated_point_cloud.reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(generated_point_cloud)
    pcd.estimate_normals()

    distances = pcd.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    radius = 3 * avg_distance

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    vertices_to_remove = densities < np.mean(densities)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    o3d.io.write_triangle_mesh("generated_bench.obj", mesh)
    o3d.visualization.draw_geometries([mesh])

if __name__ == '__main__':
    generator = Generator().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    generator.load_state_dict(torch.load('../checkpoints/generator_checkpoint.pth'))
    generate_bench_mesh(generator, 128, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
