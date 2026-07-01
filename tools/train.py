import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tol.backbones import build_backbone, build_preprocess, infer_backbone_dims, load_checkpoint
from tol.datasets import TOLDataset
from tol.models import TOLLocalizationModel


def contrastive_loss(logits_per_image, logits_per_text):
    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    return 0.5 * (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels))


def parse_args():
    parser = argparse.ArgumentParser(description="Train the main TOL localization model.")
    parser.add_argument("--stage", choices=["pr", "full"], default="full", help="Use 'pr' for retrieval-only training and 'full' for joint PR/localization training.")
    parser.add_argument("--backbone", required=True, help="clip-b16, clip-b32, or a Hugging Face SigLIP model path/name.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint used to initialize the model.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--pose-osm-dir", required=True)
    parser.add_argument("--pose-text-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/train")
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--loc-weight", type=float, default=0.1)
    parser.add_argument("--text-num", type=int, default=5)
    parser.add_argument("--pos-scale", type=float, default=25.0)
    parser.add_argument("--order", default="TNSWE")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / "tensorboard")

    backbone = build_backbone(args.backbone, args.device)
    image_dim, text_dim = infer_backbone_dims(backbone)
    model = TOLLocalizationModel(backbone, image_dim=image_dim, text_dim=text_dim, text_num=args.text_num, order=args.order).to(args.device)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, strict=False)

    dataset = TOLDataset(
        args.image_dir,
        args.text_dir,
        args.pose_osm_dir,
        args.pose_text_dir,
        city_list=args.cities,
        preprocess=build_preprocess(args.backbone, args.device),
        order=args.order,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-6, weight_decay=args.weight_decay)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for images, texts, poses_osm, poses_text, _ in tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            images = images.to(args.device)
            texts = texts.to(args.device)
            poses_osm = poses_osm.to(args.device)
            poses_text = poses_text.to(args.device)

            if args.stage == "pr":
                _, _, _, logits_per_image, logits_per_text = model.forward_retrieval(images, texts)
                loss_ret = contrastive_loss(logits_per_image, logits_per_text)
                loss_loc = torch.zeros((), device=args.device)
                loss = loss_ret
            else:
                patch_grid, _, text_desc, logits_per_image, logits_per_text = model.forward_retrieval(images, texts)
                loss_ret = contrastive_loss(logits_per_image, logits_per_text)

                top1_images = logits_per_text.argmax(dim=1)
                top1_features = patch_grid[top1_images]
                top1_poses = poses_osm[top1_images]
                _, offsets = model.forward_loc(top1_features, text_desc)

                pred_poses = top1_poses + offsets * args.pos_scale
                retrieval_errors = torch.linalg.norm(top1_poses - poses_text, dim=1)
                valid_mask = retrieval_errors <= args.pos_scale * 1.414
                loc_per_coord = F.l1_loss(pred_poses, poses_text, reduction="none")
                if valid_mask.any():
                    loss_loc = (loc_per_coord * valid_mask.unsqueeze(1).float()).sum() / valid_mask.sum()
                else:
                    loss_loc = loc_per_coord.sum() * 0.0
                loss = loss_ret + args.loc_weight * loss_loc

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss/total", loss.item(), global_step)
            writer.add_scalar("loss/retrieval", loss_ret.item(), global_step)
            writer.add_scalar("loss/localization", loss_loc.item(), global_step)
            epoch_loss += loss.item()
            global_step += 1

        ckpt_path = output_dir / f"tol_epoch_{epoch + 1:03d}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"epoch {epoch + 1}: loss={epoch_loss / max(1, len(dataloader)):.4f}, saved={ckpt_path}")


if __name__ == "__main__":
    main()
