import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        """No-op fallback when the optional tensorboard package is unavailable."""

        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass
from torchvision.models import vgg16
from pytorch_msssim import msssim

# Imports
from dataset import CloudRemovalDataset

# Models
from model1.model_convnext import Discriminator, fusion_net_depth_best
#from model1.SADT_arch import SADT
from perceptual import LossNetwork
from loss.CR_loss import ContrastLoss as crloss
from utils.metrics import psnr, ssim

NORM_MEAN = torch.tensor([0.45837133, 0.47633536, 0.44432645]).view(1, 3, 1, 1)
NORM_STD = torch.tensor([0.16936361, 0.15927625, 0.15468806]).view(1, 3, 1, 1)


def to_unit_interval(x, is_normalized):
    if not is_normalized:
        return torch.clamp(x, 0.0, 1.0)
    mean = NORM_MEAN.to(device=x.device, dtype=x.dtype)
    std = NORM_STD.to(device=x.device, dtype=x.dtype)
    return torch.clamp(x * std + mean, 0.0, 1.0)


def unit_interval_to_tanh(x):
    return x * 2.0 - 1.0


def model_output_to_unit_interval(x):
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def depth_gradient_l1(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def set_train_stage(model, stage, args):
    """
    Configures the freezing/unfreezing of model layers based on the training stage.
    """
    def set_grad(module, requires_grad):
        if module is None: return
        for param in module.parameters():
            param.requires_grad = requires_grad

    # --- Strategy for Model fusion_net_depth_best (v13) ---
    if args.model_version in [13, 23]:
        print(f"==> Setting training stage to: {stage} (fusion_net_depth_best)")
        if stage == 'depth_pretrain' and args.model_version == 23:
            # Train geometry semantics before it is allowed to affect restoration.
            set_grad(model, False)
            set_grad(model.depth_branch, True)
        elif stage == 'warmup':
            # Freeze backbones
            if hasattr(model, 'knowledge_adaptation_branch'):
                 set_grad(model.knowledge_adaptation_branch, False)
            if hasattr(model, 'depth_branch'):
                 set_grad(model.depth_branch, False)
            # Train Fusion and DWT
            set_grad(model.dwt_branch, True)
            if hasattr(model, 'fusion_router'):
                set_grad(model.fusion_router, True)
            if hasattr(model, 'depth_guided_modulation'):
                set_grad(model.depth_guided_modulation, True)
            if hasattr(model, 'base_fusion'):
                set_grad(model.base_fusion, True)
            if hasattr(model, 'depth_adapter'):
                set_grad(model.depth_adapter, True)
            set_grad(model.refine, True)
            set_grad(model.tail, True)
            set_grad(model.proj_dwt, True)
            set_grad(model.proj_ka, True)
            set_grad(model.proj_depth, True)
            
        elif stage == 'depth_warmup':
            set_grad(model, False)
            if hasattr(model, 'depth_branch'):
               set_grad(model.depth_branch, True)
            if hasattr(model, 'proj_depth'):
               set_grad(model.proj_depth, True)
            if hasattr(model, 'depth_confidence_head'):
               set_grad(model.depth_confidence_head, True)
        elif stage == 'full_finetune':
             set_grad(model, True)
        else:
             # Default to all open if unknown stage
             set_grad(model, True)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"==> Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")
        return model
        
    # Same strategy for v12 
    if args.model_version == 12:
        print(f"==> Setting training stage to: {stage} (Model v12)")
        if stage == 'warmup_decoder':
            if hasattr(model.main_model, 'encoder'):
                print(" -> Freezing Swin Encoder")
                set_grad(model.main_model.encoder, False)
            print(" -> Unfreezing DWT, Depth, AuxAdapter, Bottleneck, Decoder")
            set_grad(model.dwt_model, True)
            set_grad(model.depth_model, True)
            set_grad(model.main_model.aux_adapter, True)
            set_grad(model.main_model.bottleneck, True)
            set_grad(model.main_model.decoder, True)
            if hasattr(model.main_model, 'fusion_gate'):
                 set_grad(model.main_model.fusion_gate, True)
            if hasattr(model.main_model, 'fusion_proj'):
                 set_grad(model.main_model.fusion_proj, True)
        elif stage == 'full_finetune':
            print(" -> Unfreezing Entire Model")
            set_grad(model, True)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"==> Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")
        return model

    return model


def train_model(model, train_dataloader, test_dataloader, device, args):
    
    DNet = Discriminator().to(device)  

    model = set_train_stage(model, args.stage, args)
    active_stage = args.stage
    
    # --- Load Teacher Depth Network (RA-Depth) --- #
    encoder = None
    depth_decoder = None
    if args.use_depth:
        print("==> Loading Teacher Depth Network (RA-Depth)...")
        with torch.no_grad():
            model_path = os.path.join("./depth_teachers/ra_depth", "weights")
            if not os.path.isdir(model_path):
                print(f"Warning: Teacher depth model not found at {model_path}. Depth loss disabled.")
                args.use_depth = False
            else:
                from depth_teachers.ra_depth.networks.hrnet_encoder import hrnet18
                from depth_teachers.ra_depth.networks.depth_decoder_msf import DepthDecoder_MSF

                encoder_path = os.path.join(model_path, "encoder.pth")
                decoder_path = os.path.join(model_path, "depth.pth")
                encoder_dict = torch.load(encoder_path, map_location=device, weights_only=True)
                
                encoder = hrnet18(False)
                depth_decoder = DepthDecoder_MSF(encoder.num_ch_enc, [0], num_output_channels=1)
                
                model_dict = encoder.state_dict()
                encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
                depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
                
                encoder = encoder.to(device)
                depth_decoder = depth_decoder.to(device)
                
                encoder.eval()
                depth_decoder.eval()
                
                for param in encoder.parameters(): param.requires_grad = False
                for param in depth_decoder.parameters(): param.requires_grad = False
    
    G_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
        G_optimizer,
        milestones=args.milestones_g,
        gamma=args.lr_gamma,
    )
    D_optim = torch.optim.Adam(DNet.parameters(), lr=args.lr)
    scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
        D_optim,
        milestones=args.milestones_d,
        gamma=args.lr_gamma,
    )
    
    criterion_dehaze = nn.L1Loss() # Using simple L1 as per train5.py base
    criterion_depth = nn.L1Loss()
    msssim_loss_fn = msssim
    criterion_dehaze_cr = crloss().to(device)

    vgg_model = vgg16(pretrained=True).features[:16].to(device)
    for param in vgg_model.parameters(): param.requires_grad = False
    loss_network = LossNetwork(vgg_model).eval()

    writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))

    best_psnr = 0.0
    best_ssim = 0.0
    start_epoch = 0
    iteration = 0
    max_synth_prob = getattr(args, 'max_synth_prob', 0.5) 

    # Resume logic for Optimizer and Scheduler (if available)
    if args.resume and os.path.exists(args.resume):
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            if 'optimizer_G' in checkpoint:
                G_optimizer.load_state_dict(checkpoint['optimizer_G'])
            if 'optimizer_D' in checkpoint:
                D_optim.load_state_dict(checkpoint['optimizer_D'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'best_psnr' in checkpoint:
                best_psnr = checkpoint['best_psnr']
            if 'best_ssim' in checkpoint:
                best_ssim = checkpoint['best_ssim']
            print(f"Resumed optimizer/scheduler state from epoch {start_epoch}")
        except Exception as e:
            print(f"Could not load optimizer state: {e}. Starting optimizer from scratch.")

    print("Starting training loop (train5.py style)...")

    for epoch in range(start_epoch, args.epochs):
        if args.depth_warmup_epochs > 0:
            target_stage = 'depth_warmup' if epoch < args.depth_warmup_epochs else 'full_finetune'
        else:
            target_stage = args.stage
        if target_stage != active_stage:
            model = set_train_stage(model, target_stage, args)
            active_stage = target_stage

        model.train()
        DNet.train()

        progress = epoch / args.epochs
        current_prob = max_synth_prob * (1 - progress)
        
        if isinstance(train_dataloader.dataset, torch.utils.data.Subset):
            ds = train_dataloader.dataset.dataset
        else:
            ds = train_dataloader.dataset    
        if hasattr(ds, 'set_synthesis_prob'):
            ds.set_synthesis_prob(current_prob)

        losses = {'total_loss': [], 'img_loss': [], 'depth_loss': []}

        for batch in tqdm(train_dataloader, desc=f'Training Epoch {epoch+1}'):
            iteration += 1
            hazy = batch['cloud_img'].to(device)
            clean = batch['clear_img'].to(device)
            clean_unit = to_unit_interval(clean, args.nor)
            hazy_unit = to_unit_interval(hazy, args.nor)
            coarse_output = None
            depth_confidence = None

            # Fix for specific model versions returning tuple
            if args.model_version == 23:
                output, aux = model(hazy, return_aux=True)
                depth_pred = aux['depth']
                depth_confidence = aux['confidence']
            elif args.model_version == 13:
                res = model(hazy, return_aux=True)
                if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
                    output, aux = res
                    depth_pred = aux.get('depth', None)
                    depth_confidence = aux.get('confidence', None)
                elif isinstance(res, tuple):
                    output, depth_pred = res[0], res[1]
                else:
                    output = res
                    depth_pred = None
            elif args.model_version in [12, 14, 15]:
                res = model(hazy, return_depth=True)
                if isinstance(res, tuple):
                    output, depth_pred = res
                else:
                    output = res
                    depth_pred = None
            elif args.model_version == 16:
                output = model(hazy)
                if isinstance(output, tuple):
                    output = output[0]
                depth_pred = None
            elif args.model_version == 20:          
                res = model(hazy)
                if isinstance(res, dict):
                    output = res['dehazed']         
                    depth_pred = res.get('depth', None)
                else:
                    output = res
                    depth_pred = None
            elif args.model_version == 22:
                output, aux = model(hazy, return_aux=True)
                coarse_output = aux['coarse']
                depth_pred = None
            else:
                output = model(hazy)
                depth_pred = None

            output_unit = model_output_to_unit_interval(output)

            # --- Discriminator Step ---
            D_optim.zero_grad(set_to_none=True)
            real_out = DNet(clean_unit).mean()
            fake_out = DNet(output_unit.detach()).mean()
            D_loss = F.relu(1.0 - real_out) + F.relu(1.0 + fake_out)
            D_loss.backward()
            D_optim.step()

            # --- Generator Step ---
            G_optimizer.zero_grad(set_to_none=True)
            fake_out_new = DNet(output_unit).mean()
            adversarial_loss = -torch.mean(fake_out_new)
             
            smooth_loss_l1 = F.smooth_l1_loss(output_unit, clean_unit)
            perceptual_loss = loss_network(output_unit, clean_unit)
            msssim_loss_val = -msssim_loss_fn(output_unit, clean_unit, normalize=True)
            contrast_loss = (
                criterion_dehaze_cr(output_unit, clean_unit, hazy_unit)
                if args.contrast_weight > 0 else torch.tensor(0.0, device=device)
            )
            coarse_loss = (
                F.smooth_l1_loss(model_output_to_unit_interval(coarse_output), clean_unit)
                if coarse_output is not None
                else torch.tensor(0.0, device=device)
            )
             
            loss_total_depth = torch.tensor(0.0).to(device)
            loss_depth_abs = torch.tensor(0.0).to(device)
            loss_depth_grad = torch.tensor(0.0).to(device)
            loss_depth_conf = torch.tensor(0.0).to(device)
            if args.use_depth and encoder is not None and depth_pred is not None:
                if depth_pred.shape[1] != 1:
                    depth_pred = depth_pred[:, :1]
                with torch.no_grad():
                    real_img_2_depth_map = depth_decoder(encoder(clean_unit))[("disp", 0)]
                 
                if depth_pred.shape != real_img_2_depth_map.shape:
                    depth_pred_resized = F.interpolate(depth_pred, size=real_img_2_depth_map.shape[2:], mode='bilinear', align_corners=False)
                else:
                    depth_pred_resized = depth_pred

                depth_error = torch.abs(depth_pred_resized - real_img_2_depth_map)
                local_error = torch.abs(output_unit.detach() - clean_unit).mean(dim=1, keepdim=True)
                local_error = F.avg_pool2d(local_error, kernel_size=7, stride=1, padding=3)
                local_error = local_error / (local_error.mean(dim=(2, 3), keepdim=True) + 1e-6)
                local_error = 1.0 + 0.5 * local_error
                local_error = F.interpolate(
                    local_error,
                    size=depth_pred_resized.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )

                loss_depth_abs = 0.5 * depth_error.mean() + 0.5 * (local_error * depth_error).mean()
                loss_depth_grad = depth_gradient_l1(depth_pred_resized, real_img_2_depth_map)

                if depth_confidence is not None:
                    confidence = F.interpolate(
                        depth_confidence,
                        size=depth_pred_resized.shape[2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                    confidence_target = torch.exp(-5.0 * depth_error.detach())
                    loss_depth_conf = F.binary_cross_entropy(
                        confidence.clamp(1e-5, 1.0 - 1e-5),
                        confidence_target,
                    )

                loss_total_depth = (
                    args.depth_abs_weight * loss_depth_abs +
                    args.depth_grad_weight * loss_depth_grad +
                    args.depth_conf_weight * loss_depth_conf
                )

            total_loss = smooth_loss_l1 + \
                         args.perceptual_weight * perceptual_loss + \
                         args.adv_weight * adversarial_loss + \
                         args.msssim_weight * msssim_loss_val + \
                         args.contrast_weight * contrast_loss + \
                         args.coarse_weight * coarse_loss + \
                         args.depth_weight * loss_total_depth

            # NaN Check to prevent crash/corruption
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[Warning] NaN/Inf detected at epoch {epoch+1}. Skipping batch.")
                G_optimizer.zero_grad(set_to_none=True)
                continue
            
            # Standard Backward (No AMP)
            try:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip grad to stabilize
                G_optimizer.step()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[Warning] OOM in backward/step. Clearing cache.")
                    torch.cuda.empty_cache()
                    G_optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise e

            losses['total_loss'].append(total_loss.item())
            losses['img_loss'].append(smooth_loss_l1.item())
            if args.use_depth:
                losses['depth_loss'].append(loss_total_depth.item())

            writer.add_scalar('Loss/total', total_loss.item(), iteration)
            writer.add_scalar('Loss/img_l1', smooth_loss_l1.item(), iteration)
            writer.add_scalar('Loss/depth_abs', loss_depth_abs.item(), iteration)
            writer.add_scalar('Loss/depth_grad', loss_depth_grad.item(), iteration)
            writer.add_scalar('Loss/depth_conf', loss_depth_conf.item(), iteration)
            writer.add_scalar('Loss/adv', adversarial_loss.item(), iteration)
            writer.add_scalar('Loss/msssim', msssim_loss_val.item(), iteration)
            writer.add_scalar('Loss/perceptual', perceptual_loss.item(), iteration)

        print(f'Epoch {epoch + 1}/{args.epochs} | '
              f'Total: {np.mean(losses["total_loss"]):.4f} | '
              f'Img: {np.mean(losses["img_loss"]):.4f}')

        # Save last model periodically
        if (epoch + 1) % args.save_cycle == 0:
             torch.save(model.state_dict(), os.path.join(args.save_dir, f'dehaze_last.pth'))

        # Evaluation
        val_metrics = evaluate(model, test_dataloader, device, normalized_input=args.nor)
        writer.add_scalar('Metrics/PSNR_post', val_metrics['psnr_post'], epoch)
        writer.add_scalar('Metrics/SSIM_post', val_metrics['ssim_post'], epoch)
        writer.add_scalar('Metrics/PSNR_raw', val_metrics['psnr_raw'], epoch)
        writer.add_scalar('Metrics/SSIM_raw', val_metrics['ssim_raw'], epoch)
        print(
            f"Validation - post(PSNR/SSIM): {val_metrics['psnr_post']:.4f}/{val_metrics['ssim_post']:.4f} | "
            f"raw(PSNR/SSIM): {val_metrics['psnr_raw']:.4f}/{val_metrics['ssim_raw']:.4f}"
        )

        # Save Best PSNR/SSIM
        if val_metrics['psnr_post'] > best_psnr:
            best_psnr = val_metrics['psnr_post']
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_psnr.pth'))
            print('Saving best PSNR model...')
             
        if val_metrics['ssim_post'] > best_ssim:
            best_ssim = val_metrics['ssim_post']
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_ssim.pth'))
            print('Saving best SSIM model...')

        # Step schedulers once per epoch to match epoch-based milestones.
        scheduler_G.step()
        scheduler_D.step()
        current_lr_g = G_optimizer.param_groups[0]['lr']
        current_lr_d = D_optim.param_groups[0]['lr']
        writer.add_scalar('LR/G', current_lr_g, epoch)
        writer.add_scalar('LR/D', current_lr_d, epoch)
        print(f'LR - G: {current_lr_g:.8f}, D: {current_lr_d:.8f}')
            
        # Optional: Save checkpoint with optimizer state for resuming
        if (epoch + 1) % 500 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_G': G_optimizer.state_dict(),
                'optimizer_D': D_optim.state_dict(),
                'best_psnr': best_psnr,
                'best_ssim': best_ssim
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth'))

    writer.close()
    print('\nTrain Complete.\n')


def evaluate(model, test_dataloader, device, normalized_input=False):
    model.eval()
    psnr_post_meter = []
    ssim_post_meter = []
    psnr_raw_meter = []
    ssim_raw_meter = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating'):
            cloud_imgs = batch['cloud_img'].to(device)
            clear_imgs = batch['clear_img'].to(device)
            
            # Inference
            res = model(cloud_imgs)
            if isinstance(res, dict):
                res = res['dehazed']
            elif isinstance(res, tuple):
                res = res[0]
            if isinstance(res, tuple): res = res[0]

            pred_raw = res
            pred_post = model_output_to_unit_interval(pred_raw)
            clear_post = to_unit_interval(clear_imgs, normalized_input)
            clear_raw = unit_interval_to_tanh(clear_post)

            for i in range(len(res)):
                p_post = psnr(pred_post[i], clear_post[i])
                psnr_post_meter.append(p_post)
                s_post = ssim(pred_post[i].unsqueeze(0), clear_post[i].unsqueeze(0)).item()
                ssim_post_meter.append(s_post)

                p_raw = psnr(pred_raw[i], clear_raw[i])
                psnr_raw_meter.append(p_raw)
                s_raw = ssim(pred_raw[i].unsqueeze(0), clear_raw[i].unsqueeze(0)).item()
                ssim_raw_meter.append(s_raw)

    return {
        'psnr_post': float(np.mean(psnr_post_meter)) if psnr_post_meter else 0.0,
        'ssim_post': float(np.mean(ssim_post_meter)) if ssim_post_meter else 0.0,
        'psnr_raw': float(np.mean(psnr_raw_meter)) if psnr_raw_meter else 0.0,
        'ssim_raw': float(np.mean(ssim_raw_meter)) if ssim_raw_meter else 0.0,
    }


def main(args):
    total_dataset = CloudRemovalDataset(os.path.join(args.data_dir), args.nor, crop_size=args.crop_size)

    generator = torch.Generator().manual_seed(22)
    train_set, test_set = random_split(
        total_dataset,
        [int(len(total_dataset) * 0.9), len(total_dataset) - int(len(total_dataset) * 0.9)],
        generator=generator
    )

    train_sampler = None
    if hasattr(total_dataset, 'sample_weights'):
        train_weights = torch.as_tensor(
            [total_dataset.sample_weights[i] for i in train_set.indices],
            dtype=torch.double
        )
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(test_set, batch_size=1, # BS=1 for accurate metrics during eval
                             shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f'Using device: {device}')

    if args.model_version == 12:
        model = fusion_net12().to(device)
    elif args.model_version == 1:
        model = fusion_net_1().to(device)
    elif args.model_version == 13:
        # fusion_net_depth_best
        model = fusion_net_depth_best(
            crop_size=args.crop_size,
            semantic_backbone=args.semantic_backbone
        ).to(device)
    elif args.model_version == 14:
        # fusion_net_depth_best
        model = fusion_net_depth_best_res(crop_size=args.crop_size, backbone='resnet34').to(device)
    elif args.model_version == 15:
        # fusion_net_depth_best
        from model1.model_vssm import fusion_net_depth_best_mamba
        model = fusion_net_depth_best_mamba(crop_size=args.crop_size, backbone='visionmamba').to(device)
    elif args.model_version == 16:
        # fusion_net_depth_best
        model = SADT(in_channels=3,window_size=8,use_bias=True,reduction=4,out_channels=3).to(device)
    elif args.model_version == 17:
        # fusion_net_depth_best
        model = fusion_net_depth_best_light(crop_size=args.crop_size).to(device)
    elif args.model_version == 18:
        # fusion_net_depth_best
        model = fusion_net_depth_best_v2(crop_size=args.crop_size).to(device)
    elif args.model_version == 19:
        # fusion_net_depth_best
        model = fusion_net_depth_best_v3(crop_size=args.crop_size).to(device)
    elif args.model_version == 20:
        # fusion_net_depth_best
        model = SADGDehazeV3(pretrained=True).to(device)
    elif args.model_version == 21:
        model = fusion_net_real_lite(crop_size=args.crop_size).to(device)
    elif args.model_version == 22:
        model = fusion_net_real_lite_v2(crop_size=args.crop_size).to(device)
    elif args.model_version == 23:
        model = fusion_net_depth_geometry_v1(crop_size=args.crop_size).to(device)
    else:
        # Fallback list or logic
        if args.model_version == 11: model = fusion_net11().to(device)
        else: model = fusion_net_depth_best(crop_size=args.crop_size).to(device)

    if args.resume is not None:
        if os.path.exists(args.resume):
            print(f'Resume from {args.resume}')
            # Loading weights flexibly
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f'Checkpoint {args.resume} not found, starting from scratch.')

    train_model(model, train_loader, test_loader, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DehazeXL Train V5 Style (Restored)')
    parser.add_argument('--data_dir', type=str, default=r"/newhome/zhangbaoguo/project1/Dense-HAZE/train/",
                        help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default=r'./checkpoints/dense',
                        help='Path to save checkpoints')
    parser.add_argument('--save_cycle', type=int, default=100,
                        help='Cycle of saving checkpoint (epoch)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Epochs')
    parser.add_argument('--nor', action='store_true',
                        help='Normalize the image')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='Crop size')
    parser.add_argument('--model_version', type=int, choices=[13], default=13,
                        help='Only version 13 (fusion_net_depth_best) is retained')
    parser.add_argument('--stage', type=str, default='full_finetune',
                        choices=['warmup', 'depth_warmup', 'full_finetune'],
                        help='Training stage')
    parser.add_argument('--depth_warmup_epochs', type=int, default=0,
                        help='If >0, run depth_warmup for N epochs then switch to full_finetune')
    parser.add_argument('--depth_weight', type=float, default=0.1,
                        help='Weight for depth loss')
    parser.add_argument('--depth_abs_weight', type=float, default=1.0,
                        help='Weight for absolute depth consistency loss')
    parser.add_argument('--depth_grad_weight', type=float, default=0.5,
                        help='Weight for depth gradient consistency loss')
    parser.add_argument('--depth_conf_weight', type=float, default=0.1,
                        help='Weight for depth confidence calibration loss')
    parser.add_argument('--semantic_backbone', type=str, default='lightweight_v2',
                        choices=['convnext', 'lightweight', 'lightweight_v2'],
                        help='Semantic branch backbone for fusion_net_depth_best')
    parser.add_argument('--use_depth', dest='use_depth', action='store_true',
                        help='Use depth network')
    parser.add_argument('--no_use_depth', dest='use_depth', action='store_false',
                        help='Disable depth network')
    parser.set_defaults(use_depth=True)

    parser.add_argument('--perceptual_weight', type=float, default=0.01,
                        help='Weight for perceptual loss')
    parser.add_argument('--adv_weight', type=float, default=0.0005,
                        help='Weight for adversarial loss')
    parser.add_argument('--msssim_weight', type=float, default=0.2,
                        help='Weight for MS-SSIM loss')
    parser.add_argument('--contrast_weight', type=float, default=0.0,
                        help='Weight for contrastive loss (0 disables it)')
    parser.add_argument('--coarse_weight', type=float, default=0.2,
                        help='Weight for V22 coarse restoration supervision')
    parser.add_argument('--milestones_g', type=int, nargs='+', default=[3000, 5000, 8000],
                        help='Epoch milestones for generator LR decay')
    parser.add_argument('--milestones_d', type=int, nargs='+', default=[5000, 7000, 8000],
                        help='Epoch milestones for discriminator LR decay')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='LR decay factor for MultiStepLR')
    parser.add_argument('--max_synth_prob', type=float, default=0.5,
                        help='Maximum synthesis probability for data augmentation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    main(args)
