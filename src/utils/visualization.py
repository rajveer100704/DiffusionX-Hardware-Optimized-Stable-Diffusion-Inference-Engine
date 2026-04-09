import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Visualizer:
    """
    Utilities for saving images and creating denoising progression visualizations.
    """
    def __init__(self, config):
        self.output_dir = config.paths.images
        self.prog_dir = config.paths.progression
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.prog_dir, exist_ok=True)

    def save_image(self, image_np, filename):
        """
        Saves a single image (numpy array [H, W, 3]).
        """
        image_np = (image_np * 255).astype(np.uint8)
        img = Image.fromarray(image_np)
        path = os.path.join(self.output_dir, filename)
        img.save(path)
        return path

    def save_grid(self, images_np, filename, rows=1):
        """
        Saves a grid of images.
        """
        # images_np is [B, H, W, 3]
        num_images = images_np.shape[0]
        cols = (num_images + rows - 1) // rows
        
        h, w, c = images_np.shape[1:]
        grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
        
        for i in range(num_images):
            r, c_idx = i // cols, i % cols
            img = (images_np[i] * 255).astype(np.uint8)
            grid[r*h : (r+1)*h, c_idx*w : (c_idx+1)*w, :] = img
            
        img = Image.fromarray(grid)
        path = os.path.join(self.output_dir, filename)
        img.save(path)
        return path

    def create_progression_gif(self, frames, filename, duration=100):
        """
        Creates a GIF from a list of frames (latents-decoded).
        """
        path = os.path.join(self.prog_dir, filename)
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        return path

    @torch.no_grad()
    def decode_latents(self, vae, latents):
        """
        Helper to decode intermediate latents for visualization.
        """
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).astype(np.uint8)
        return Image.fromarray(image)
