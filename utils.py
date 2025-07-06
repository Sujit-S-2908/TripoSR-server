import numpy as np
import trimesh
from PIL import Image

def remove_background(image: Image.Image, foreground_ratio: float = 0.85) -> Image.Image:
    # Dummy background removal (real version requires rembg)
    return image

def save_mesh_as_glb(verts, faces, path: str):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(path)
