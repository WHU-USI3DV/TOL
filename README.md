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
- 2026-04-03: [Project page](https://whu-usi3dv.github.io/TOL/) (with introduction video) is aviliable!🎉 Data, code and models will be released. 

## 💡 Citation
If you find this repo helpful, please give us a star~.Please consider citing Mobile-Seed if this program benefits your project.
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
- [CVG-Text](https://github.com/yejy53/CVG-Text) for text-to-OSM/satellite image retrievlal;
- [FreeReg](https://github.com/WHU-USI3DV/FreeReg) for excellent template;