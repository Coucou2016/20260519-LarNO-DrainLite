# Shenzhen Region1 — ITZI Flood Simulation

基于 LarNO 论文深圳 Region1 数据集（20m 分辨率）的 ITZI 二维洪涝模拟分析。

## 数据来源

从 `E:\Projects\20260519-LarNO` **复制**（非移动），原项目内容未修改：

| 数据 | 来源路径 |
|------|----------|
| DEM 地形 | `LarNO-main/benchmark/urbanflood/geodata/region1_20m/dem.npy` |
| 降雨/参考水深 | `LarNO-main/benchmark/urbanflood/flood/region1_20m/event*/` |
| OSM 管网 | `extended_study/output/osm_merged_network.npz` |
| SWMM 管网 | `extended_study/output/drainage_network.inp` |

## 目录结构

```
shenzhen_region1/
├── input_data/
│   ├── geodata/region1_20m/dem.npy
│   ├── flood/region1_20m/event{1,20,65-70}/
│   └── networks/osm_merged_network.npz
├── output/                    # 模拟结果
├── visualization_output/      # 可视化图表
├── run_itzi_shenzhen.py       # 运行模拟
└── visualize_shenzhen.py      # 生成可视化
```

## 运行

### 简化管网排水（orifice 近似）
```bash
python run_itzi_shenzhen.py
python visualize_shenzhen.py
```

### ITZI + SWMM 完整耦合（仅对比 MIKE+ 参考）
```bash
python build_swmm_from_osm.py              # 从 OSM 主干网生成 SWMM inp
python run_full_domain_coupled.py --domain sub    # 子区域 4×5.6 km
python run_full_domain_coupled.py --domain full   # 全域 8×11.2 km
python visualize_full_coupled.py
```

耦合实现参考 `test_cases/urban_drainage/run_coupled_simulation.py`：
`DrainageSimulation` + `apply_coupling_to_nodes` 双向交换地表水深与管网流量。

## 模拟设置

- 子区域：4km × 5.6km（与 LarNO extended_study 一致）
- 网格：20m 分辨率
- 事件：event1, event20, event65–event70（8 个 MIKE+ 参考事件）
- 情景 A：仅地表径流（ITZI Surface）
- 情景 B：地表 + OSM 简易管网排水（ITZI + Pipes）
