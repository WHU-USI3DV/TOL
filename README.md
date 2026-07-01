<h2> 
<a href="https://whu-usi3dv.github.io/TOL/" target="_blank">TOL: Textual Localization with OpenStreetMap</a>
</h2>

This is the official PyTorch implementation of the following publication:

> **TOL: Textual Localization with OpenStreetMap**<br/>
> [Youqi Liao](https://martin-liao.github.io/), [Shuhao Kang](https://kang-1-2-3.github.io/), [Jingyu Xu](https://kevinxu-01.github.io/home/), [Olaf Wysocki](https://olafwysocki.github.io/), [Yan Xia](https://yan-xia.github.io/), [Jianping Li](https://jianping.xyz/), [Zhen Dong](https://dongzhenwhu.github.io/index.html), [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm),[Xieyuanli Chen](https://chen-xieyuanli.github.io)<br/>
> *Tech Repo*<br/>
> **Paper** | [**Arxiv**](https://arxiv.org/abs/2604.01644) | [**Project-page**](https://whu-usi3dv.github.io/TOL/) | [**Video**](https://youtu.be/Sz2qG8KKEgw)


## 🔭 Introduction
<p align="center">
<strong>TL;DR: TOL is a large-scale text-to-OSM localization benchmark. To address this task, we propose TOLoc, a corresponding localization pipeline.</strong>
</p>
<img src="./illustration.png" alt="Motivation" style="zoom:50%;">

<p align="justify">
<strong>Abstract:</strong> Natural language provides an intuitive way to express spatial intent in geospatial applications. While existing localization methods often rely on dense point cloud maps or high-resolution imagery, OpenStreetMap (OSM) offers a compact and freely available map representation that encodes rich semantic and structural information, making it well suited for large-scale localization. However, text-to-OSM (T2O) localization remains largely unexplored. In this paper, we formulate the T2O global localization task, which aims to estimate accurate 2 degree-of-freedom (DoF) positions in urban environments from textual scene descriptions without relying on geometric observations or GNSS-based initial location. To support the proposed task, we introduce TOL, a large-scale benchmark spanning multiple continents and diverse urban environments. TOL contains approximately 121K textual queries paired with OSM map tiles and covers about 316 km of road trajectories across Boston, Karlsruhe, and Singapore. We further propose TOLoc, a coarse-to-fine localization framework that explicitly models the semantics of surrounding objects and their directional information. In the coarse stage, direction-aware features are extracted from both textual descriptions and OSM tiles to construct global descriptors, which are used to retrieve candidate locations for the query. In the fine stage, the query text and top-1 retrieved tile are jointly processed, where a dedicated alignment module fuses textual descriptor and local map features to regress the 2-DoF pose. Experimental results demonstrate that TOLoc achieves strong localization performance, outperforming the best existing method by 6.53%, 9.93%, and 8.31% at 5m, 10m, and 25m thresholds, respectively, and shows strong generalization to unseen environments. 
</p>

## 🆕 News
- 2026-04-03: [Project page](https://whu-usi3dv.github.io/TOL/) (with introduction video) is available!🎉 Data, code and models will be released.
- 2026-07-01: Data and code are released.

## 🛠️ Installation
Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/WHU-USI3DV/TOL.git
cd TOL
conda create -n tol python=3.10 -y
conda activate tol
pip install -r requirements.txt
```


## 📦 Data Preparation
The training and evaluation code expects each TOL split to contain four aligned folders:

```text
data/TOL-N or data/TOL-K360
├── images/<city_or_sequence>/*.png
├── texts/<city_or_sequence>/*.txt
├── poses_osm/<city_or_sequence>/*.txt
└── poses_text/<city_or_sequence>/*.txt
```

Each sample is matched by file stem. For example, `images/boston-seaport/xxx.png`, `texts/boston-seaport/xxx.txt`, `poses_osm/boston-seaport/xxx.txt`, and `poses_text/boston-seaport/xxx.txt` describe the same query-map pair.

### Use the Released TOL Data
After downloading the released data archive from [Baidu disk](https://pan.baidu.com/s/12GidMmlwnY3LrvWTOWGwDA?pwd=sd9z) or [Google Drive](https://drive.google.com/file/d/1GlLkqyahxRrZTxV_vw1rsDc5sdxYRs5z/view?usp=sharing), place it in the repository root and extract it:

```bash
unzip TOL_data.zip -d .
```

The expected folders are:

```text
data/TOL-N/images
data/TOL-N/texts
data/TOL-N/poses_osm
data/TOL-N/poses_text
data/TOL-K360/images
data/TOL-K360/texts
data/TOL-K360/poses_osm
data/TOL-K360/poses_text
```

### Regenerate OSM Tiles and Text Queries
We also provide scripts to regenerate OSM raster tiles and rule-based textual descriptions from raw nuScenes or KITTI-360 poses. The OSM files used by the scripts are stored under `maploc/data/`.

For TOL-N, prepare the nuScenes trainval data first, then run:

```bash
NUSCENES_ROOT=/path/to/nuscenes \
MODE=nuscenes \
bash scripts/prepare_tol_data.sh
```

For TOL-K360, prepare KITTI-360 pose folders first, then run:

```bash
K360_POSE_ROOT=/path/to/KITTI-360/data_poses \
MODE=kitti360 \
bash scripts/prepare_tol_data.sh
```

Useful options include `TILE_SIZE=50`, `PPM=4`, `TEXT_RADIUS=100`, and `RUN_TEXT=1`. The script writes generated tiles, raster images, OSM poses, and text descriptions into `data/TOL-N` or `data/TOL-K360`. If you regenerate data from scratch, make sure the folders passed to training are aligned with the four-folder layout shown above.

## 🚀 Training
The recommended training schedule first trains the place-recognition branch and then initializes the full localization model from the PR checkpoint.

Example for TOL-N with CLIP ViT-B/16:

```bash
IMAGE_DIR=data/TOL-N/images \
TEXT_DIR=data/TOL-N/texts \
POSE_OSM_DIR=data/TOL-N/poses_osm \
POSE_TEXT_DIR=data/TOL-N/poses_text \
BACKBONE=clip-b16 \
BATCH_SIZE=32 \
PR_EPOCHS=30 \
LOC_EPOCHS=30 \
OUTPUT_ROOT=outputs/toln_clip_b16 \
bash scripts/train_pr_then_loc.sh
```

Example for TOL-K360:

```bash
IMAGE_DIR=data/TOL-K360/images \
TEXT_DIR=data/TOL-K360/texts \
POSE_OSM_DIR=data/TOL-K360/poses_osm \
POSE_TEXT_DIR=data/TOL-K360/poses_text \
BACKBONE=clip-b16 \
OUTPUT_ROOT=outputs/tolk360_clip_b16 \
bash scripts/train_pr_then_loc.sh
```

To train on selected cities or sequences only, pass a space-separated `CITIES` list:

```bash
CITIES="boston-seaport singapore-onenorth" \
IMAGE_DIR=data/TOL-N/images \
TEXT_DIR=data/TOL-N/texts \
POSE_OSM_DIR=data/TOL-N/poses_osm \
POSE_TEXT_DIR=data/TOL-N/poses_text \
bash scripts/train_pr_then_loc.sh
```

The script saves retrieval checkpoints to `${OUTPUT_ROOT}/pr/tol_epoch_*.pth`, full localization checkpoints to `${OUTPUT_ROOT}/full/tol_epoch_*.pth`, and TensorBoard logs under each output directory.

## 🔍 Evaluation
Use `scripts/evaluate_tol.sh` to evaluate any trained or released TOLoc checkpoint. Change `CHECKPOINT`, `BACKBONE`, and `SAVE_DIR` for the model you want to test:

```bash
CHECKPOINT=weights/loc/TOL-C-B16.pth \
IMAGE_DIR=data/TOL-N/images \
TEXT_DIR=data/TOL-N/texts \
POSE_OSM_DIR=data/TOL-N/poses_osm \
POSE_TEXT_DIR=data/TOL-N/poses_text \
BACKBONE=clip-b16 \
SAVE_DIR=outputs/eval_TOL-C-B16 \
bash scripts/evaluate_tol.sh
```

Released checkpoints could be downloaded from [Baidu Disk](https://pan.baidu.com/s/15_-juY3iJPT2jh9YeLb0Sg?pwd=66xt) or [Google Drive](https://drive.google.com/drive/folders/1rE9Wwrr0s0ztxIKsIdyg7SKr0ob32ONg?usp=sharing):

| Model | Checkpoint | BACKBONE |
| --- | --- | --- |
| TOLoc-C-B32 | `weights/loc/TOL-C-B32.pth` | `clip-b32` |
| TOLoc-C-B16 | `weights/loc/TOL-C-B16.pth` | `clip-b16` |
| TOLoc-S-B224 | `weights/loc/TOL-S-B224.pth` | `google/siglip-base-patch16-224` |
| TOLoc-S-B384 | `weights/loc/TOL-S-B384.pth` | `google/siglip-base-patch16-384` |

`C` denotes CLIP-based models, and `S` denotes SigLIP-based models. Supported CLIP aliases are `clip-b16` and `clip-b32`; for SigLIP checkpoints, pass the matching Hugging Face model name or a local model path as `BACKBONE`.

The script prints two kinds of results. Retrieval recall, such as `R@1_10m`, means the percentage of text queries whose top-1 retrieved OSM tile is within 10 m of the ground-truth text pose. Localization success, such as `Top-1 localization success @5m`, means the percentage of queries whose final predicted position is within 5 m. Intermediate descriptors, top-k retrieval indices, and per-query localization results are saved under `SAVE_DIR`, including `loc_results.pkl`.

The released models should give Top@1 localization results close to:

| Model | R@1 | R@5 | R@10 | Loc@5m | Loc@10m | Loc@25m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TOLoc-C-B32 | 7.21 | 17.01 | 24.63 | 4.07 | 6.58 | 26.44 |
| TOLoc-C-B16 | 8.72 | 18.68 | 26.55 | 3.51 | 5.38 | 18.53 |
| TOLoc-S-B224 | 6.42 | 17.30 | 26.82 | 4.35 | 6.56 | 18.59 |
| TOLoc-S-B384 | 6.83 | 18.65 | 28.10 | 4.30 | 6.43 | 15.75 |

## 💡 Citation
If you find this repo helpful, please give us a star~. Please consider citing TOL if this program benefits your project.
```
@misc{liao2026toltextuallocalizationopenstreetmap,
      title={TOL: Textual Localization with OpenStreetMap}, 
      author={Youqi Liao and Shuhao Kang and Jingyu Xu and Olaf Wysocki and Yan Xia and Jianping Li and Zhen Dong and Bisheng Yang and Xieyuanli Chen},
      year={2026},
      eprint={2604.01644},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.01644}, 
}
```

## 🔗 Related Projects
We sincerely thank the excellent projects:
- [GOTPR](https://github.com/donghwijung/GOTPR) for graph-based text-to-OSM place recognition;
- [CVG-Text](https://github.com/yejy53/CVG-Text) for text-to-OSM/satellite image retrieval;
- [FreeReg](https://github.com/WHU-USI3DV/FreeReg) for excellent template;
