from PIL import Image
import torch
import os
from models.model import TripoSR
from torchvision import transforms
import numpy as np
import trimesh
import mcubes
from utils import remove_background, save_mesh_as_glb

class TripoSRModel:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = TripoSR()
        self.model.to(self.device)
        checkpoint = torch.load("checkpoints/tripoSR.ckpt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.eval()

    def run(self, image_path: str, output_dir: str = "outputs") -> str:
        os.makedirs(output_dir, exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        image = remove_background(image, foreground_ratio=0.85)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            voxel_output = self.model(image_tensor)
        voxel_grid = voxel_output[0].squeeze().cpu().numpy()
        verts, faces = mcubes.marching_cubes(voxel_grid, 0)
        out_path = os.path.join(output_dir, "tripoSR_output.glb")
        save_mesh_as_glb(verts, faces, out_path)
        return out_path
