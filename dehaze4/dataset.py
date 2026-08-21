import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from haze_synthesizer import HazeSynthesizer  # Import synthesizer


class CloudRemovalDataset(Dataset):
    def __init__(self, root_dir, normalize, crop_size):
        self.root_dir = root_dir
        self.clear_dir = os.path.join(self.root_dir, 'GT')
        self.depth_dir = os.path.join(self.root_dir, 'Depth')  # Depth map directory
        self.image_filenames = [img for img in os.listdir(self.clear_dir) if
                                img.endswith('.png') or img.endswith('.jpg')]
        transforms_list = [
            transforms.ToTensor()
        ]
        self.crop_size = crop_size
        self.normalize = normalize
        if normalize:
            self.norm_transform = transforms.Normalize([0.45837133, 0.47633536, 0.44432645],
                                                        [0.16936361, 0.15927625, 0.15468806])
        else:
            self.norm_transform = None

        self.transform = transforms.Compose(transforms_list) # Only ToTensor initially

        # Dynamic Augmentation Parameters
        self.synthesis_prob = 0.0
        self.synthesizer = HazeSynthesizer()

        if not os.path.exists(self.depth_dir):
             print(f"Warning: Depth directory {self.depth_dir} not found. Synthesis will be disabled (fallback to real hazy images).")

    def set_synthesis_prob(self, prob):
        self.synthesis_prob = prob

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # 1. Decide whether to synthesize
        do_synthesis = False
        if self.synthesis_prob > 0 and random.random() < self.synthesis_prob:
            do_synthesis = True

        clear_img_name = os.path.join(self.clear_dir, self.image_filenames[idx])
        clear_img = Image.open(clear_img_name).convert("RGB")

        # Load real hazy image if not synthesizing OR if synthesis fails?
        # Actually standard flow is load both, but if we synthesize, we replace cloud_img.
        # However, to be efficient, we might skip loading real hazy if synthesizing.
        # But we need consistent cropping.

        # Load real hazy image anyway to get dimensions and handle verify size consistency
        cloud_dir = os.path.join(self.root_dir, 'hazy'.format(random.choice([1, 2, 3, 4]))) # .format bug in original code
        cloud_img_name = os.path.join(cloud_dir, self.image_filenames[idx])
        try:
            cloud_img = Image.open(cloud_img_name).convert("RGB")
        except FileNotFoundError:
            # Fallback if real hazy doesn't exist but we want to synthesize?
            # For now assume dataset integrity.
            cloud_img = clear_img.copy() # Placeholder

        if cloud_img.size != clear_img.size:
            raise ValueError("The size of the input image pairs is inconsistent")

        img_width, img_height = clear_img.size

        if img_width >= self.crop_size and img_height >= self.crop_size:
            # 图像尺寸足够，进行随机裁剪
            start_x = random.randint(0, img_width - self.crop_size)
            start_y = random.randint(0, img_height - self.crop_size)
            clear_img = clear_img.crop((start_x, start_y, start_x + self.crop_size, start_y + self.crop_size))
            cloud_img = cloud_img.crop((start_x, start_y, start_x + self.crop_size, start_y + self.crop_size))

            # Crop Depth if needed
            crop_coords = (start_x, start_y, start_x + self.crop_size, start_y + self.crop_size)
        else:
            # 图像尺寸过小，调整到裁剪尺寸
            print(
                f"Warning: Resizing image {self.image_filenames[idx]} from {img_width}x{img_height} to {self.crop_size}x{self.crop_size}")
            clear_img = clear_img.resize((self.crop_size, self.crop_size), Image.BICUBIC)
            cloud_img = cloud_img.resize((self.crop_size, self.crop_size), Image.BICUBIC)
            crop_coords = None # Flag that full image was resized

        # Second crop in original code?
        # Original code has a second crop block after resizing/first crop.
        # "start_x = random.randint(0, clear_img.size[0] - self.crop_size)"
        # This seems redundant if first block ran, but handles resized case.
        # Let's preserve original logic flow but intercept for depth and synthesis.

        start_x = random.randint(0, clear_img.size[0] - self.crop_size)
        start_y = random.randint(0, clear_img.size[1] - self.crop_size)
        clear_img = clear_img.crop((start_x, start_y, start_x + self.crop_size, start_y + self.crop_size))
        cloud_img = cloud_img.crop((start_x, start_y, start_x + self.crop_size, start_y + self.crop_size))

        # Capture final crop coordinates if we need to crop depth map
        # Since original code does multiple crops, tracking is hard.
        # But we only need depth map crop to match FINAL clear_img crop.
        # The issue is we don't know the exact cumulative transform unless we replicate it on depth.

        # Strategy: Load depth map, apply same transforms (Crop, Resize, Crop).
        # To do this cleanly, we need to restructure the crop logic to apply to both.
        # Or just apply to depth here if synthesis is active.

        depth_img = None
        if do_synthesis:
            base_name = os.path.splitext(self.image_filenames[idx])[0]
            depth_name = os.path.join(self.depth_dir, base_name + '.png') # Assuming png depth
            if os.path.exists(depth_name):
                depth_img = Image.open(depth_name) # Depth usually single channel
                # Resize/Crop sequence
                # 1. Resize/Crop 1
                if img_width >= self.crop_size and img_height >= self.crop_size:
                     if crop_coords: # First crop coords
                         depth_img = depth_img.crop(crop_coords)
                else:
                     depth_img = depth_img.resize((self.crop_size, self.crop_size), Image.BICUBIC)

                # 2. Crop 2
                depth_img = depth_img.crop((start_x, start_y, start_x + self.crop_size, start_y + self.crop_size))
            else:
                # Missing depth map, fallback to real hazy or synthesize without depth?
                # Synthesis requires depth. Fallback to real hazy.
                do_synthesis = False

        random_rotate = random.choice([0, 90, 180, 270])
        clear_img = clear_img.rotate(random_rotate)
        cloud_img = cloud_img.rotate(random_rotate)
        if do_synthesis and depth_img:
            depth_img = depth_img.rotate(random_rotate)

        # Convert to Tensor
        clear_tensor = transforms.ToTensor()(clear_img)
        # cloud_tensor = transforms.ToTensor()(cloud_img) # Don't convert yet if replacing

        if do_synthesis and depth_img:
             depth_tensor = transforms.ToTensor()(depth_img) # [1, H, W]
             # Generate synthetic hazy
             # clear_tensor is [0, 1]
             # synthesizer returns [1, C, H, W], we need [C, H, W] for dataset
             cloud_tensor = self.synthesizer.synthesize(clear_tensor, depth_tensor).squeeze(0)
             # cloud_tensor is [0, 1]
        else:
             cloud_tensor = transforms.ToTensor()(cloud_img)

        # Apply Normalization
        if self.norm_transform:
            clear_tensor = self.norm_transform(clear_tensor)
            cloud_tensor = self.norm_transform(cloud_tensor)

        sample = {'cloud_img': cloud_tensor, 'clear_img': clear_tensor, 'is_synthesized': do_synthesis}
        return sample


if __name__ == '__main__':
    dataset = CloudRemovalDataset(r"./datasets/8KDehaze_mini", False, crop_size=2048)
    print(dataset[0])
