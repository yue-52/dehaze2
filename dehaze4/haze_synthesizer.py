import torch
import random

class HazeSynthesizer:
    def __init__(self):
        pass

    def synthesize(self, clear_img_tensor, depth_map_tensor):
        """
        Synthesize hazy image from clear image and depth map.

        Args:
            clear_img_tensor: Normalize tensor [C, H, W] or [1, C, H, W], range likely [0, 1] or normalized.
                              The dataset currently normalizes the image.
                              We need to denormalize if we want to apply physics-based synthesis correctly,
                              or assume input is in [0, 1].
            depth_map_tensor: Tensor [1, H, W], normalized depth (or disparity).

        Formula:
            I(x) = (J(x)^gamma + N) * t(x) + (A + dA) * (1 - t(x))
            t(x) = exp(-beta * d(x))

            gamma ~ U[1.3, 1.7]
            beta ~ U[0.8, 1.7]
            A ~ U[0.8, 1.0]
            dA ~ U[-0.025, 0.025] -> applied to each channel? The text says vector.
            N ~ Gaussian noise.
        """

        device = clear_img_tensor.device
        
        # Ensure inputs are [B, C, H, W]
        if clear_img_tensor.dim() == 3:
            clear_img_tensor = clear_img_tensor.unsqueeze(0)
        if depth_map_tensor.dim() == 3:
            depth_map_tensor = depth_map_tensor.unsqueeze(0)

        # Assumption: clear_img_tensor is in [0, 1] range.
        # If dataset normalizes, we might need to handle that.
        # Looking at dataset.py, it normalizes with mean/std.
        # We should perform synthesis on Un-normalized data for physical correctness,
        # then re-normalize.

        bs, c, h, w = clear_img_tensor.shape
        
        # Sample parameters per image in batch
        gamma = 1.3 + (1.7 - 1.3) * torch.rand(bs, 1, 1, 1, device=device)
        beta = 0.8 + (1.7 - 0.8) * torch.rand(bs, 1, 1, 1, device=device)
        A = 0.8 + (1.0 - 0.8) * torch.rand(bs, 1, 1, 1, device=device)
        dA = -0.025 + 0.05 * torch.rand(bs, 3, 1, 1, device=device) # 3 channel vector

        # Apply Gamma Correction
        # J(x)^gamma
        # Avoid NaN if J(x) < 0 (due to normalization).
        # So we MUST denormalize first if input is normalized.

        J = clear_img_tensor

        # Transmission t(x) = exp(-beta * d(x))
        # depth map should be in some reasonable range for this formula to work.
        # usually depth map is normalized 0 to 1 or disparity.
        t = torch.exp(-beta * depth_map_tensor)
        
        # Noise
        N = torch.randn_like(J) * 0.01 # The text mentions Gaussian noise N, but doesn't specify std dev. I'll assume small.
                                       # Actually text says "introduced Gaussian noise distribution N".
                                       # Usually N(0, sigma). I'll use sigma=0.01 as a guess or keep it essentially small.

        # Formula: I = (J^gamma + N) * t + (A + dA) * (1 - t)

        J_pow = torch.pow(J + 1e-6, gamma) # safety for 0

        term1 = (J_pow + N) * t
        term2 = (A + dA) * (1 - t)

        I = term1 + term2

        # Clip to valid range [0, 1] if necessary
        I = torch.clamp(I, 0, 1)
        
        return I

# Define normalization means and stds from dataset.py to help denormalizing
MEAN = [0.45837133, 0.47633536, 0.44432645]
STD = [0.16936361, 0.15927625, 0.15468806]

def denormalize(tensor):
    mean = torch.tensor(MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

def normalize(tensor):
    mean = torch.tensor(MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std

