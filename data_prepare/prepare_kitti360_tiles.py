import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from maploc.osm.tiling import BoundaryBox, TileManager
from maploc.osm.viz import Colormap
from maploc.utils.geo import Projection


EARTH_RADIUS = 6378137.0
KITTI360_ORIGIN = [48.9843445, 8.4295857]


def latlon_to_mercator(lat, lon, scale):
    mx = scale * lon * np.pi * EARTH_RADIUS / 180.0
    my = scale * EARTH_RADIUS * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    return mx, my


def mercator_to_latlon(mx, my, scale):
    lon = mx * 180.0 / (scale * np.pi * EARTH_RADIUS)
    lat = 360.0 / np.pi * np.arctan(np.exp(my / (scale * EARTH_RADIUS))) - 90.0
    return lat, lon


def lat_to_scale(lat):
    return np.cos(lat * np.pi / 180.0)


def load_poses(path):
    data = np.loadtxt(path)
    timestamps = data[:, 0].astype(np.int32)
    poses = np.reshape(data[:, 1:], (-1, 3, 4))
    bottom = np.tile(np.array([0, 0, 0, 1]).reshape(1, 1, 4), (poses.shape[0], 1, 1))
    return timestamps, np.concatenate((poses, bottom), axis=1)


def postprocess_poses(poses):
    transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return [(transform @ pose.T).T for pose in poses]


def pose_to_latlon(pose):
    scale = lat_to_scale(KITTI360_ORIGIN[0])
    ox, oy = latlon_to_mercator(KITTI360_ORIGIN[0], KITTI360_ORIGIN[1], scale)
    origin = np.array([ox, oy, 0.0])
    translation = pose[:3, 3] + origin
    return mercator_to_latlon(translation[0], translation[1], scale)


def export_latlon_files(pose_root):
    pose_root = Path(pose_root)
    for seq_dir in tqdm(sorted(p for p in pose_root.iterdir() if p.is_dir()), desc="latlon"):
        _, poses = load_poses(seq_dir / "poses.txt")
        latlons = [pose_to_latlon(pose) for pose in postprocess_poses(poses)]
        with (seq_dir / "latlon.txt").open("w", encoding="utf-8") as f:
            for lat, lon in latlons:
                f.write(f"{lat:.6f} {lon:.6f}\n")


def prepare_tiles(pose_root, osm_path, output_root, tile_size=50, ppm=4):
    pose_root = Path(pose_root)
    output_root = Path(output_root)
    tile_dir = output_root / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    for seq_dir in tqdm(sorted(p for p in pose_root.iterdir() if p.is_dir()), desc="tiles"):
        latlon = np.loadtxt(seq_dir / "latlon.txt")[:, :2]
        projection = Projection.from_points(latlon)
        xy = projection.project(latlon)
        bbox_min = np.floor(xy.min(0) / tile_size) * tile_size
        bbox_max = np.ceil(xy.max(0) / tile_size) * tile_size
        bbox = BoundaryBox(bbox_min, bbox_max) + tile_size
        manager = TileManager.from_bbox(projection, bbox, ppm, path=Path(osm_path), tile_size=tile_size)
        manager.save(tile_dir / f"tiles_{seq_dir.name}_{tile_size}_{ppm}.pkl")


def export_samples(pose_root, output_root, tile_size=50, ppm=4, save_raster=True):
    pose_root = Path(pose_root)
    output_root = Path(output_root)
    tile_dir = output_root / "tiles"
    raster_dir = output_root / "raster" / f"raster_osm_{tile_size}_{ppm}"
    color_dir = output_root / "raster_color" / f"raster_osm_color_{tile_size}_{ppm}"
    pose_dir = output_root / "poses_osm" / f"poses_osm_{tile_size}_{ppm}"

    for seq_dir in sorted(p for p in pose_root.iterdir() if p.is_dir()):
        latlon = np.loadtxt(seq_dir / "latlon.txt")[:, :2]
        manager = TileManager.load(tile_dir / f"tiles_{seq_dir.name}_{tile_size}_{ppm}.pkl")
        for idx, point in enumerate(tqdm(latlon, desc=seq_dir.name)):
            xy = manager.projection.project(point)
            bbox = BoundaryBox(xy - tile_size / 2, xy + tile_size / 2)
            canvas = manager.query(bbox)
            if save_raster:
                raster = canvas.raster.transpose((1, 2, 0))
                raster_color = (Colormap.apply(canvas.raster) * 255).astype(np.uint8)[..., ::-1]
                (raster_dir / seq_dir.name).mkdir(parents=True, exist_ok=True)
                (color_dir / seq_dir.name).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(raster_dir / seq_dir.name / f"{idx:06d}.png"), raster)
                cv2.imwrite(str(color_dir / seq_dir.name / f"{idx:06d}.png"), raster_color)
            (pose_dir / seq_dir.name).mkdir(parents=True, exist_ok=True)
            (pose_dir / seq_dir.name / f"{idx:06d}.txt").write_text(f"{xy[0]}, {xy[1]}\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare KITTI-360 OSM raster tiles for TOL.")
    parser.add_argument("--pose-root", required=True)
    parser.add_argument("--osm-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tile-size", type=int, default=50)
    parser.add_argument("--ppm", type=int, default=4)
    parser.add_argument("--skip-raster", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_latlon_files(args.pose_root)
    prepare_tiles(args.pose_root, args.osm_path, args.output_root, args.tile_size, args.ppm)
    export_samples(args.pose_root, args.output_root, args.tile_size, args.ppm, save_raster=not args.skip_raster)
