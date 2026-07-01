import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from maploc.osm.parser import Groups


def make_polar_grid(radius_samples: int, angle_samples: int):
    radius = np.arange(1, radius_samples + 1)
    angles = np.arange(angle_samples) / angle_samples * 2 * np.pi - 0.25 * np.pi
    directions = np.stack([np.cos(angles), -np.sin(angles)], axis=1)
    return radius.reshape(-1, 1, 1) * directions[np.newaxis, ...] + radius_samples


def polar_resample(raster, grid):
    height, width, _ = raster.shape
    grid = np.round(grid).astype(int)
    grid[..., 0] = np.clip(grid[..., 0], 0, width - 1)
    grid[..., 1] = np.clip(grid[..., 1], 0, height - 1)
    return raster[grid[..., 1], grid[..., 0], :]


def find_objects_before_building(raster):
    result = {"areas": [], "ways": [], "nodes": []}
    for angle_idx in range(raster.shape[1]):
        building_mask = raster[:, angle_idx, 0] == 1
        stop_idx = int(np.argmax(building_mask)) if building_mask.any() else raster.shape[0]
        for channel, key in zip([0, 1, 2], ["areas", "ways", "nodes"]):
            values = raster[: stop_idx + 1, angle_idx, channel]
            result[key].extend(values[values > 0].tolist())
    return {key: np.unique(values).astype(np.uint8) for key, values in result.items()}


def object_names(values):
    descriptions = []
    for key, values_for_key in values.items():
        if key == "nodes":
            lookup = Groups.nodes
        elif key == "ways":
            lookup = Groups.ways
        else:
            lookup = Groups.areas
        for value in values_for_key:
            descriptions.append(lookup[value - 1])
    return descriptions


def build_descriptions(elements, max_per_direction=None):
    lines = []
    for direction, features in elements.items():
        prefix = "The pose is on top of:" if direction == "top" else f"The pose is {direction} of:"
        features = list(features)
        if max_per_direction is not None and len(features) > max_per_direction:
            features = random.sample(features, max_per_direction)
        content = " ".join(features) if features else "None"
        lines.append(f"{prefix} {content}")
    return lines


def generate_texts(raster_dir, output_dir, cities, radius=100, angle_samples=360, max_per_direction=None):
    raster_dir = Path(raster_dir)
    output_dir = Path(output_dir)
    grid = make_polar_grid(radius, angle_samples)

    for city in cities:
        city_input = raster_dir / city
        city_output = output_dir / city
        city_output.mkdir(parents=True, exist_ok=True)
        for image_path in tqdm(sorted(city_input.glob("*.png")), desc=city):
            raster = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            polar = polar_resample(raster, grid)
            q = angle_samples // 4
            sectors = {
                "east": polar[:, :q],
                "north": polar[:, q : 2 * q],
                "west": polar[:, 2 * q : 3 * q],
                "south": polar[:, 3 * q :],
            }
            elements = {direction: object_names(find_objects_before_building(sector)) for direction, sector in sectors.items()}
            lines = build_descriptions(elements, max_per_direction=max_per_direction)
            (city_output / f"{image_path.name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text descriptions from rasterized OSM tiles.")
    parser.add_argument("--raster-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cities", nargs="+", required=True)
    parser.add_argument("--radius", type=int, default=100)
    parser.add_argument("--angle-samples", type=int, default=360)
    parser.add_argument("--max-per-direction", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_texts(
        args.raster_dir,
        args.output_dir,
        args.cities,
        radius=args.radius,
        angle_samples=args.angle_samples,
        max_per_direction=args.max_per_direction,
    )
