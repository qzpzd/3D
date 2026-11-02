# 工业 3D 点云处理基线

这个项目把工业点云常见的传统处理链和深度学习方法组织为一套可运行、可测试的最小工程：

```text
原始点云
→ ROI/直通滤波 → 体素下采样 → 统计/半径去噪
→ RANSAC 平面与工装移除 → DBSCAN 聚类
→ 目标工件选择 → 法线/边缘 → OBB/AABB 精细裁剪
→ 尺寸测量 / FPFH-RANSAC-ICP 配准 / PatchCore-3D 异常检测
```

它面向研究、算法选型和数据链路验证，不是已经完成现场验收的质检软件。深度演示使用合成正常样本训练，结果不能代替真实产线的漏检率、误报率或计量能力评估。

## 快速运行

在 PowerShell 中执行：

```powershell
cd C:\Users\18562\Documents\ChatGPT\3d
python -m pip install -r requirements.txt
python -m pytest -q
python -m industrial_pc3d.cli demo --output outputs/demo
python -m industrial_pc3d.cli registration-demo --output outputs/registration_demo
python -m industrial_pc3d.cli deep-demo --output outputs/deep_demo --epochs 3
```

安装并启用 Open3D 后端：

```powershell
python -m pip install -r requirements-open3d.txt
python -m industrial_pc3d.cli process-open3d `
  --input outputs\demo\scene_raw.pcd `
  --config configs\traditional.json `
  --output outputs\open3d_demo
```

合成工位演示会输出：

- `outputs/demo/target.pcd`：选中的工件点云。
- `outputs/demo/target_analysis.npz`：坐标、局部曲率/边缘分数。
- `outputs/demo/run_summary.json`：各阶段点数、耗时、平面、簇和 OBB。
- `outputs/demo/pipeline_preview.png`：原始场景、聚类和目标边缘预览。

深度演示会输出 `patchcore3d.pt`、`evaluation_summary.json` 和 `anomaly_preview.png`。默认只训练 6 轮，CPU 可运行；结果用于验证代码路径和指标计算。

## 代码结构

| 文件 | 内容 |
|---|---|
| `industrial_pc3d/io.py` | ASCII/二进制 PCD、ASCII PLY、XYZ/NPY/NPZ 读取；点云保存 |
| `industrial_pc3d/geometry.py` | 直通、体素、SOR/半径去噪、RANSAC 平面、DBSCAN、PCA-OBB、AABB、边缘、Kabsch |
| `industrial_pc3d/pipeline.py` | 从原始场景到目标工件的可配置传统流水线 |
| `industrial_pc3d/open3d_backend.py` | Open3D 滤波、聚类、OBB、FPFH-RANSAC-ICP 后端 |
| `industrial_pc3d/registration.py` | FPFH 风格描述子、特征 RANSAC、点到点 ICP |
| `industrial_pc3d/deep_models.py` | PointNet 分类/分割、PointNet++ 风格集合抽象、局部几何编码器 |
| `industrial_pc3d/patchcore3d.py` | 仅正常样本训练、coreset 记忆库、点级异常定位 |
| `industrial_pc3d/synthetic.py` | 工位、遮挡/噪声、凹坑/凸包合成数据 |
| `industrial_pc3d/cli.py` | 全部可运行入口 |
| `configs/` | 传统与深度示例参数 |
| `cpp/` | PCL C++ 流水线与 CMake 构建配置 |
| `docs/工业3D点云方法与应用.md` | 方法选型、研究依据、工业风险与评价体系 |
| `docs/运行手册.md` | 输入、命令、参数、排错与现场迁移步骤 |
| `docs/库后端与代码流程说明.md` | 自研/调库对比、图像语义、安装、PCL 构建、深度数据入口 |

## 真实数据命令

单场景目标提取：

```powershell
python -m industrial_pc3d.cli process `
  --input data\scan.pcd `
  --config configs\traditional.json `
  --output outputs\scan_001
```

模板配准：

```powershell
python -m industrial_pc3d.cli align `
  --source data\scan.pcd `
  --target data\template.pcd `
  --output outputs\alignment_001
```

所有长度使用同一单位；示例单位为米。现场若是毫米，必须同步修改体素、ROI、平面阈值、聚类半径和 OBB 边距，避免把 0.5 mm 当作 0.5 m。

## 生产化边界

- 测量任务优先使用标定后的扫描坐标、CAD/模板配准和尺寸不确定度分析；不要用深层特征取代可追溯的几何量。
- 缺少训练缺陷时应先采用无监督异常检测；有足够像素/点级缺陷标签后再比较 PointNet++、DGCNN、RandLA-Net 等监督分割模型。
- 量产阈值必须由独立正常/异常集和产线节拍数据确定，并监控 FPR、漏检、P99 延迟、点密度和环境漂移。
- 代码中的 PCA-OBB 快速稳定，但并非所有形状的严格最小体积 OBB；高精度外形测量应使用 CAD 或专用最小包络求解器。
- PatchCore-3D 演示描述子由训练得到的局部几何编码和显式 PCA/径向统计拼接而成，是教学基线，不是官方 PatchCore 的 3D 复刻。

详细依据和取舍见 [方法与应用](docs/工业3D点云方法与应用.md)，执行步骤见 [运行手册](docs/运行手册.md)。

三种传统处理后端的详细区别、依赖安装和完整输入输出见
[库后端与代码流程说明](docs/库后端与代码流程说明.md)。

