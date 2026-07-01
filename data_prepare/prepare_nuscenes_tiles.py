import argparse
from pathlib import Path

import cv2
import numpy as np
from nuscenes import NuScenes
from tqdm import tqdm

from maploc.osm.tiling import BoundaryBox, TileManager
from maploc.osm.viz import Colormap
from maploc.utils.geo import Projection, TopocentricConverter


NUSCENES_ORIGINS = {
    "boston-seaport": [42.336849169438615, -71.05785369873047, 0],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984, 0],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555, 0],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758, 0],
}


def collect_poses(nusc):
    converters = {name: TopocentricConverter(*origin) for name, origin in NUSCENES_ORIGINS.items()}
    poses = {name: {"latlon": [], "xy": [], "token": []} for name in NUSCENES_ORIGINS}
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        log = nusc.get("log", scene["log_token"])
        location = log["location"]
        sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        xy = np.array(ego_pose["translation"][:2])
        latlon = converters[location].to_lla(xy[0], xy[1], 0)[:2]
        poses[location]["latlon"].append(latlon)
        poses[location]["xy"].append(xy)
        poses[location]["token"].append(sample["token"])
    return poses


def prepare_tiles(poses, osm_dir, output_root, tile_size=50, ppm=4):
    osm_dir = Path(osm_dir)
    output_root = Path(output_root)
    tile_dir = output_root / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    for city, data in poses.items():
        latlon = np.array(data["latlon"])
        projection = Projection.from_points(latlon)
        xy = projection.project(latlon)
        bbox_min = np.floor(xy.min(0) / tile_size) * tile_size
        bbox_max = np.ceil(xy.max(0) / tile_size) * tile_size
        bbox = BoundaryBox(bbox_min, bbox_max) + tile_size
        manager = TileManager.from_bbox(projection, bbox, ppm, path=osm_dir / f"{city}.osm", tile_size=tile_size)
        manager.save(tile_dir / f"tiles_{city}_{tile_size}_{ppm}.pkl")


def export_samples(poses, output_root, tile_size=50, ppm=4):
    output_root = Path(output_root)
    tile_dir = output_root / "tiles"
    raster_dir = output_root / "raster" / f"raster_osm_{tile_size}_{ppm}"
    color_dir = output_root / "raster_color" / f"raster_osm_color_{tile_size}_{ppm}"
    pose_dir = output_root / "poses_osm" / f"poses_osm_{tile_size}_{ppm}"

    for city, data in poses.items():
        manager = TileManager.load(tile_dir / f"tiles_{city}_{tile_size}_{ppm}.pkl")
        for latlon, token in tqdm(zip(data["latlon"], data["token"]), total=len(data["latlon"]), desc=city):
            xy = manager.projection.project(latlon)
            bbox = BoundaryBox(xy - tile_size / 2, xy + tile_size / 2)
            canvas = manager.query(bbox)
            raster = canvas.raster.transpose((1, 2, 0))
            raster_color = (Colormap.apply(canvas.raster) * 255).astype(np.uint8)[..., ::-1]

            (raster_dir / city).mkdir(parents=True, exist_ok=True)
            (color_dir / city).mkdir(parents=True, exist_ok=True)
            (pose_dir / city).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(raster_dir / city / f"{token}.png"), raster)
            cv2.imwrite(str(color_dir / city / f"{token}.png"), raster_color)
            (pose_dir / city / f"{token}.txt").write_text(f"{xy[0]}, {xy[1]}\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare NuScenes OSM raster tiles for TOL.")
    parser.add_argument("--nuscenes-root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--osm-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tile-size", type=int, default=50)
    parser.add_argument("--ppm", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=True)
    all_poses = collect_poses(nusc)
    prepare_tiles(all_poses, args.osm_dir, args.output_root, args.tile_size, args.ppm)
    export_samples(all_poses, args.output_root, args.tile_size, args.ppm)
