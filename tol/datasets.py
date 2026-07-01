from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


ORDER_TOKEN_TO_TEXT = {
    "T": ("top",),
    "N": ("north",),
    "S": ("south",),
    "E": ("east",),
    "W": ("west",),
    "NE": ("northeast",),
    "SE": ("southeast",),
    "SW": ("southwest",),
    "NW": ("northwest",),
}


def parse_order_tokens(order):
    if isinstance(order, str) and "," in order:
        return [token.strip() for token in order.split(",") if token.strip()]
    return list(order)


def read_pose(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return np.array([float(v) for v in f.readline().strip().split(",")], dtype=np.float32)


class TOLDataset(Dataset):
    def __init__(
        self,
        image_dir,
        text_dir,
        pose_osm_dir,
        pose_text_dir,
        city_list=None,
        preprocess=None,
        order="TNSWE",
        text_all_in_one=False,
    ):
        self.image_dir = Path(image_dir)
        self.text_dir = Path(text_dir)
        self.pose_osm_dir = Path(pose_osm_dir)
        self.pose_text_dir = Path(pose_text_dir)
        self.preprocess = preprocess
        self.order = order
        self.text_all_in_one = text_all_in_one

        cities = city_list or sorted(p.name for p in self.image_dir.iterdir() if p.is_dir())
        self.samples = []
        for city in cities:
            images = sorted((self.image_dir / city).glob("*.png"))
            for image_path in tqdm(images, desc=f"load {city}"):
                stem = image_path.stem
                text_path = self.text_dir / city / f"{stem}.txt"
                if not text_path.exists():
                    legacy_text_path = self.text_dir / city / f"{image_path.name}.txt"
                    text_path = legacy_text_path if legacy_text_path.exists() else text_path
                pose_osm_path = self.pose_osm_dir / city / f"{stem}.txt"
                pose_text_path = self.pose_text_dir / city / f"{stem}.txt"
                if text_path.exists() and pose_osm_path.exists() and pose_text_path.exists():
                    self.samples.append((image_path, text_path, pose_osm_path, pose_text_path))

        if not self.samples:
            raise RuntimeError("No samples found. Check input directories and city names.")

    def _read_text(self, path: Path):
        if self.text_all_in_one:
            return path.read_text(encoding="utf-8").strip()

        lines = path.read_text(encoding="utf-8").splitlines()
        order_tokens = parse_order_tokens(self.order)
        direction_lines = {}
        for line in lines:
            line_lower = line.lower()
            for token in order_tokens:
                for phrase in ORDER_TOKEN_TO_TEXT.get(token, ()):
                    matched = "top of" in line_lower if phrase == "top" else f"{phrase} of" in line_lower
                    if matched:
                        direction_lines[token] = line
                        break

        output = []
        for token in order_tokens:
            if token in direction_lines:
                output.append(direction_lines[token])
            else:
                phrase = ORDER_TOKEN_TO_TEXT.get(token, (token,))[0]
                output.append("The pose is on top of: None" if phrase == "top" else f"The pose is {phrase} of: None")
        return output

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, text_path, pose_osm_path, pose_text_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        text = self._read_text(text_path)
        if self.preprocess is not None:
            image, text = self.preprocess(image, text)
        pose_osm = read_pose(pose_osm_path)
        pose_text = read_pose(pose_text_path)
        return image, text, pose_osm, pose_text, image_path.stem


def collate_tol_batch(batch):
    images, texts, poses_osm, poses_text, names = zip(*batch)
    if torch.is_tensor(images[0]):
        images = torch.stack(images, dim=0)
    if torch.is_tensor(texts[0]):
        texts = torch.stack(texts, dim=0)
    poses_osm = torch.from_numpy(np.stack(poses_osm)).float()
    poses_text = torch.from_numpy(np.stack(poses_text)).float()
    return images, texts, poses_osm, poses_text, list(names)
