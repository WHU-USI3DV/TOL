import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tol.backbones import build_backbone, build_preprocess, infer_backbone_dims, load_checkpoint
from tol.datasets import TOLDataset
from tol.models import TOLLocalizationModel


def ensure_feature_dirs(save_dir: Path):
    for name in ["image_feat", "image_emb", "text", "poses_osm", "poses_text"]:
        (save_dir / name).mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def eval_retrieval(model, dataloader, device, save_dir, pose_thresholds=(10.0, 25.0)):
    model.eval()
    save_dir = Path(save_dir)
    ensure_feature_dirs(save_dir)

    image_descs, text_descs, poses_osm, poses_text, manifest = [], [], [], [], []
    for images, texts, pose_osm, pose_text, names in tqdm(dataloader, desc="retrieval"):
        images = images.to(device)
        texts = texts.to(device)
        patch_grid, image_desc, text_desc, _, _ = model.forward_retrieval(images, texts)

        for idx, name in enumerate(names):
            manifest.append(name)
            torch.save(patch_grid[idx : idx + 1].cpu(), save_dir / "image_feat" / f"{name}.pt")
            torch.save(image_desc[idx : idx + 1].cpu(), save_dir / "image_emb" / f"{name}.pt")
            torch.save(text_desc[idx : idx + 1].cpu(), save_dir / "text" / f"{name}.pt")
            torch.save(pose_osm[idx : idx + 1].cpu(), save_dir / "poses_osm" / f"{name}.pt")
            torch.save(pose_text[idx : idx + 1].cpu(), save_dir / "poses_text" / f"{name}.pt")

        image_descs.append(image_desc.cpu())
        text_descs.append(text_desc.cpu())
        poses_osm.append(pose_osm.cpu())
        poses_text.append(pose_text.cpu())

    np.save(save_dir / "manifest.npy", np.array(manifest))
    image_descs = torch.cat(image_descs, dim=0)
    text_descs = torch.cat(text_descs, dim=0)
    poses_osm = torch.cat(poses_osm, dim=0).numpy()
    poses_text = torch.cat(poses_text, dim=0).numpy()

    similarity = text_descs @ image_descs.t()
    ranks = similarity.argsort(dim=1, descending=True).numpy()
    np.savez(save_dir / "topk_indices.npz", ranks=ranks)

    metrics = {}
    for threshold in pose_thresholds:
        for k in [1, 5, 10]:
            hits = 0
            for idx in range(len(ranks)):
                dists = np.linalg.norm(poses_text[idx : idx + 1] - poses_osm, axis=1)
                positives = np.where(dists <= threshold)[0]
                hits += np.isin(ranks[idx, :k], positives).any()
            metric_name = f"R@{k}_{threshold:g}m"
            metrics[metric_name] = hits / len(ranks) * 100.0
            print(f"{metric_name}: {metrics[metric_name]:.2f}")
    return metrics


@torch.no_grad()
def eval_localization(model, save_dir, device, topk=10, pos_scale=25.0, success_thresholds=(5.0, 10.0, 25.0), percentiles=(5.0, 10.0, 25.0)):
    save_dir = Path(save_dir)
    manifest = list(np.load(save_dir / "manifest.npy"))
    if not manifest:
        raise RuntimeError(f"No samples found in {save_dir / 'manifest.npy'}.")
    ranks_all = np.load(save_dir / "topk_indices.npz")["ranks"]
    topk = min(topk, ranks_all.shape[1])
    ranks = ranks_all[:, :topk]

    all_errors, all_center_errors, loc_results = [], [], {}
    for i, name in tqdm(list(enumerate(manifest)), desc="localization"):
        text = torch.load(save_dir / "text" / f"{name}.pt").to(device)
        pose_text = torch.load(save_dir / "poses_text" / f"{name}.pt").to(device)

        candidate_names = [manifest[idx] for idx in ranks[i]]
        image_features = torch.cat([torch.load(save_dir / "image_feat" / f"{n}.pt") for n in candidate_names], dim=0).to(device)
        poses_osm = torch.cat([torch.load(save_dir / "poses_osm" / f"{n}.pt") for n in candidate_names], dim=0).to(device)

        _, offsets = model.forward_loc(image_features, text.repeat(image_features.shape[0], 1))
        pose_pred = poses_osm + offsets * pos_scale
        errors = torch.linalg.norm(pose_pred - pose_text, dim=1)
        center_errors = torch.linalg.norm(poses_osm - pose_text, dim=1)
        all_errors.append(errors.cpu().numpy())
        all_center_errors.append(center_errors.cpu().numpy())
        loc_results[name] = {
            "top1_map": candidate_names[0],
            "pose_text": pose_text.cpu().numpy(),
            "pose_pred": pose_pred[0].cpu().numpy(),
            "pose_map": poses_osm[0].cpu().numpy(),
            "error": errors[0].cpu().item(),
        }

    errors = np.array(all_errors)
    center_errors = np.array(all_center_errors)
    for k in [1, 3, 5, 10]:
        if k > topk:
            continue
        best = errors[:, :k].min(axis=1)
        center_best = center_errors[:, :k].min(axis=1)
        success = " / ".join(f"@{threshold:g}m {(best <= threshold).mean() * 100:.2f}" for threshold in success_thresholds)
        center_success = " / ".join(f"@{threshold:g}m {(center_best <= threshold).mean() * 100:.2f}" for threshold in success_thresholds)
        error_percentiles = np.percentile(best, percentiles)
        percentile_text = " / ".join(f"P{percentile:g} {value:.2f}m" for percentile, value in zip(percentiles, error_percentiles))
        print(f"Top-{k} localization success: {success}")
        print(f"Top-{k} center success: {center_success}")
        print(f"Top-{k} localization error percentiles: {percentile_text}")

    with (save_dir / "loc_results.pkl").open("wb") as f:
        pickle.dump(loc_results, f)
    return loc_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TOL localization models.")
    parser.add_argument("--backbone", required=True, help="clip-b16, clip-b32, or a Hugging Face SigLIP model path/name.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--pose-osm-dir", required=True)
    parser.add_argument("--pose-text-dir", required=True)
    parser.add_argument("--save-dir", default="outputs/eval")
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-num", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--pos-scale", type=float, default=25.0)
    parser.add_argument("--order", default="TNSWE")
    parser.add_argument("--recall-thresholds", nargs="+", type=float, default=[10.0, 25.0])
    parser.add_argument("--success-thresholds", nargs="+", type=float, default=[5.0, 10.0, 25.0])
    parser.add_argument("--error-percentiles", nargs="+", type=float, default=[5.0, 10.0, 25.0])
    return parser.parse_args()


def main():
    args = parse_args()
    backbone = build_backbone(args.backbone, args.device)
    image_dim, text_dim = infer_backbone_dims(backbone)
    model = TOLLocalizationModel(backbone, image_dim=image_dim, text_dim=text_dim, text_num=args.text_num, order=args.order).to(args.device)
    load_checkpoint(model, args.checkpoint)

    dataset = TOLDataset(
        args.image_dir,
        args.text_dir,
        args.pose_osm_dir,
        args.pose_text_dir,
        city_list=args.cities,
        preprocess=build_preprocess(args.backbone, args.device),
        order=args.order,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    eval_retrieval(model, dataloader, args.device, args.save_dir, pose_thresholds=args.recall_thresholds)
    eval_localization(
        model,
        args.save_dir,
        args.device,
        topk=args.topk,
        pos_scale=args.pos_scale,
        success_thresholds=args.success_thresholds,
        percentiles=args.error_percentiles,
    )


if __name__ == "__main__":
    main()
