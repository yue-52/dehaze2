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
        self.image_filenames = sorted(
            [
                img for img in os.listdir(self.clear_dir)
                if img.endswith('.png') or img.endswith('.jpg')
            ]
        )
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
        self.to_tensor = transforms.ToTensor()

        # Dynamic Augmentation Parameters
        self.synthesis_prob = 0.0
        self.synthesizer = HazeSynthesizer()
        self.beta_bands = [
            (0.80, 1.10),
            (1.10, 1.40),
            (1.40, 1.70),
        ]
        self.scene_buckets = ["dark", "mid", "bright"]
        self.scene_bucket_by_index = []
        self.sample_weights = []
        self._build_scene_balance_weights()

        if not os.path.exists(self.depth_dir):
             print(f"Warning: Depth directory {self.depth_dir} not found. Synthesis will be disabled (fallback to real hazy images).")

    def _classify_scene_bucket(self, image_path):
        with Image.open(image_path).convert("RGB") as img:
            thumb = img.resize((64, 64), Image.BICUBIC)
            arr = np.asarray(thumb, dtype=np.float32) / 255.0
            luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            mean_l = float(luminance.mean())
        if mean_l < 0.33:
            return "dark"
        if mean_l > 0.66:
            return "bright"
        return "mid"

    def _build_scene_balance_weights(self):
        bucket_counts = {k: 0 for k in self.scene_buckets}
        for fname in self.image_filenames:
            clear_path = os.path.join(self.clear_dir, fname)
            bucket = self._classify_scene_bucket(clear_path)
            self.scene_bucket_by_index.append(bucket)
            bucket_counts[bucket] += 1

        for bucket in self.scene_bucket_by_index:
            count = max(bucket_counts[bucket], 1)
            self.sample_weights.append(1.0 / float(count))

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

        # Load real hazy image for paired alignment or synthesis fallback.
        cloud_dir = os.path.join(self.root_dir, 'hazy')
        cloud_img_name = os.path.join(cloud_dir, self.image_filenames[idx])
        try:
            cloud_img = Image.open(cloud_img_name).convert("RGB")
        except FileNotFoundError:
            cloud_img = clear_img.copy()

        if cloud_img.size != clear_img.size:
            raise ValueError("The size of the input image pairs is inconsistent")

        img_width, img_height = clear_img.size

        depth_img = None
        depth_path = None
        if do_synthesis:
            base_name = os.path.splitext(self.image_filenames[idx])[0]
            depth_path = os.path.join(self.depth_dir, base_name + '.png')
            if os.path.exists(depth_path):
                depth_img = Image.open(depth_path)
            else:
                do_synthesis = False

        if img_width < self.crop_size or img_height < self.crop_size:
            print(
                f"Warning: Resizing image {self.image_filenames[idx]} from {img_width}x{img_height} to {self.crop_size}x{self.crop_size}")
            clear_img = clear_img.resize((self.crop_size, self.crop_size), Image.BICUBIC)
            cloud_img = cloud_img.resize((self.crop_size, self.crop_size), Image.BICUBIC)
            if do_synthesis and depth_img is not None:
                depth_img = depth_img.resize((self.crop_size, self.crop_size), Image.BICUBIC)
            crop_box = (0, 0, self.crop_size, self.crop_size)
        else:
            start_x = random.randint(0, img_width - self.crop_size)
            start_y = random.randint(0, img_height - self.crop_size)
            crop_box = (start_x, start_y, start_x + self.crop_size, start_y + self.crop_size)
            clear_img = clear_img.crop(crop_box)
            cloud_img = cloud_img.crop(crop_box)
            if do_synthesis and depth_img is not None:
                depth_img = depth_img.crop(crop_box)

        if clear_img.size != (self.crop_size, self.crop_size):
            clear_img = clear_img.crop(crop_box)
        if cloud_img.size != (self.crop_size, self.crop_size):
            cloud_img = cloud_img.crop(crop_box)

        random_rotate = random.choice([0, 90, 180, 270])
        clear_img = clear_img.rotate(random_rotate)
        cloud_img = cloud_img.rotate(random_rotate)
        if do_synthesis and depth_img is not None:
            depth_img = depth_img.rotate(random_rotate)

        # Convert to Tensor
        clear_tensor = self.to_tensor(clear_img)

        beta_band = None
        if do_synthesis and depth_img is not None:
             depth_tensor = self.to_tensor(depth_img)
             beta_band = self.beta_bands[idx % len(self.beta_bands)]
             cloud_tensor = self.synthesizer.synthesize(
                 clear_tensor,
                 depth_tensor,
                 beta_range=beta_band,
             ).squeeze(0)
        else:
             cloud_tensor = self.to_tensor(cloud_img)

        # Apply Normalization
        if self.norm_transform:
            clear_tensor = self.norm_transform(clear_tensor)
            cloud_tensor = self.norm_transform(cloud_tensor)

        sample = {
            'cloud_img': cloud_tensor,
            'clear_img': clear_tensor,
            'is_synthesized': do_synthesis,
            'scene_bucket': self.scene_bucket_by_index[idx],
            'beta_band': beta_band if beta_band is not None else (0.0, 0.0),
        }
        return sample


if __name__ == '__main__':
    dataset = CloudRemovalDataset(r"./datasets/8KDehaze_mini", False, crop_size=2048)
    print(dataset[0])
