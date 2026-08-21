import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms

from depth_teachers.ra_depth import networks
from depth_teachers.ra_depth.layers import disp_to_depth

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate depth maps for dataset using RA-Depth.')
    
    parser.add_argument('--data_dir', type=str, required=True,default="/newhome/zhangbaoguo/project1/Dense-HAZE/train/GT/",
                        help='Path to dataset root (containing GT folder)')
    parser.add_argument('--model_name', type=str, default='RA-Depth',
                        help='name of a pretrained model to use')
    parser.add_argument('--no_cuda', action='store_true', help='if set, disables CUDA')
    
    return parser.parse_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Paths
    gt_dir = os.path.join(args.data_dir, 'GT')
    depth_out_dir = os.path.join(args.data_dir, 'Depth')
    
    if not os.path.exists(gt_dir):
        print(f"Error: GT directory not found at {gt_dir}")
        return

    if not os.path.exists(depth_out_dir):
        os.makedirs(depth_out_dir)

    # Load Model
    model_path = os.path.join("depth_teachers", "ra_depth", "weights")
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    print("Loading model from", model_path)
    # Assuming hrnet18 based on test_simple.py in RA-Depth
    encoder = networks.hrnet18(False) 
    # Use DepthDecoder_MSF and scales=range(1) as in test_simple.py
    depth_decoder = networks.DepthDecoder_MSF(num_ch_enc=encoder.num_ch_enc, scales=range(1))

    # Load weights
    try:
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return

    encoder.to(device)
    encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()

    # Process Images
    image_paths = glob.glob(os.path.join(gt_dir, '*.png')) + glob.glob(os.path.join(gt_dir, '*.jpg'))
    print(f"Found {len(image_paths)} images.")

    with torch.no_grad():
        for i, image_path in enumerate(image_paths):
            input_image = pil.open(image_path).convert('RGB')
            # Resize? test_simple.py doesn't resize strictly but feeds feed_height?
            # Actually test_simple.py loads feed_height from model weights.
            # But RA-Depth/HRNet might expect specific size?
            # For simplicity, we feed original resolution or resize to model's training res?
            # Helper in test_simple.py:
            # feed_height = loaded_dict_enc['height']
            # feed_width = loaded_dict_enc['width']
            # input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            
            # Let's inspect test_simple.py logic again.
            # It loads height/width from encoder dict.
            feed_height = loaded_dict_enc['height']
            feed_width = loaded_dict_enc['width']
            original_size = input_image.size
            input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            
            input_tensor = transforms.ToTensor()(input_image_resized).unsqueeze(0).to(device)
            
            features = encoder(input_tensor)
            outputs = depth_decoder(features)
            
            disp = outputs[("disp", 0)]
            pred_disp, _ = disp_to_depth(disp, 0.1, 100) # Min/Max depth
            
            # Upsample to original size?
            pred_disp = torch.nn.functional.interpolate(
                pred_disp, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False)
            
            # Convert Disparity to Depth
            # Depth = 1 / Disparity
            # Avoid division by zero
            pred_depth = 1.0 / (pred_disp + 1e-6)
            
            # Normalize Depth for saving? 
            # Or save as float .npy? Or save as 16-bit PNG?
            # dataset.py uses Image.open(). 16-bit PNG is best for depth image.
            # But PIL handling of I;16 is tricky.
            # Simple approach: Save normalized 8-bit assuming relative depth is enough for synthesis.
            # Synthesis formula t = exp(-beta * d).
            # If d is absolute metric depth, beta controls visibility distance.
            # If d is normalized [0, 1], beta needs to be adjusted.
            # The paper says d(x) is depth map. Beta in [0.8, 1.7].
            # This beta range suggests d(x) is likely in range [0, ~10] or similar low range?
            # If d is 100 meters, exp(-1.7 * 100) is 0.
            # If d is normalized [0, 1], exp(-1.7 * 1) = 0.18. exp(-0.8 * 1) = 0.44.
            # This looks like d should be somewhat normalized.
            # Let's normalize depth map to [0, 1] per image? Or globally?
            # Per image normalization is safer for relative depth.
            
            depth_min = pred_depth.min()
            depth_max = pred_depth.max()
            pred_depth_norm = (pred_depth - depth_min) / (depth_max - depth_min + 1e-6)
            
            # Save as 8-bit PNG
            pred_depth_np = (pred_depth_norm.squeeze().cpu().numpy() * 255).astype(np.uint8)
            depth_pil = pil.fromarray(pred_depth_np)
            
            filename = os.path.basename(image_path)
            # Ensure png extension for depth
            save_name = os.path.splitext(filename)[0] + '.png'
            depth_pil.save(os.path.join(depth_out_dir, save_name))
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(image_paths)}")

    print("Done. Depth maps saved to", depth_out_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)

