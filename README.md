# MaD-PF: Modular and Differentiable Particle Filter for Cross-Environment SE(3) LiDAR Localization

Implementation for a differentiable particle filter for SE(3) robot localization using a 3D LIDAR.
Multiple modules are available for the observation model, motion model and resampling strategies.

**Motion models** - [`dynamics.py`](./diff_slam/dynamics.py):
- Odometry only model with `OdometrySE3Config`
- RealNVP from TODO: add citation with `RealNVPConfig`
- Anchored-Ensembles RealNVP model with `EnsembleConfig`
- Anchored-Ensembles MLP model with `MLPConfig`

**Observation models** - [`obs.py`](./diff_slam/obs.py):
- Pillar-Encoder based with `PPObserverConfig` (both with and without pooling) 
- Neural Signed Distance Function based with `SDFOBSModuleConfig`

**Resampling strategies** - [`resample.py`](./diff_slam/resample.py):
- Contraction-based with `ContractResampleConfig`
- Gumbel Softmax with `GSResampleConfig`
- Optimal-Transport based with `OTResampleConfig`

This project also contains an ICP-based SLAM implementation to generate reference trajectories that is used to generate dataset labels from raw ROS2 recordings. 

## Installation

1. Install uv
2. Create a virtual environment with `uv venv`
3. Install dependencies with `uv sync`
4. Download the training data
TODO: remove data as it is probably not double blind
```bash
mkdir -p data
# Original Jackal Data
curl -L "https://drive.usercontent.google.com/download?id=1Qjf40SJPy6MPzpXJtT0DS3HKSPe45upO&confirm=t" | tar -xf - -C data
# TUM RoboRacer/F1TENTH Data
curl -L "https://drive.usercontent.google.com/download?id=10_hjd2Vy9NJ_pMslh-IomGywuB03HzpH&confirm=t" | tar -xf - -C data
# ETF Jackal Data 
curl -L "https://drive.usercontent.google.com/download?id=1L84A4_gUZ6srrGiLp1a5TU74fSg3WYXQ&confirm=t" | tar -xzf - -C data
# ETH Mocap Environment internal Jackal Rosbags and Mocap GT Data 
curl -L "https://drive.usercontent.google.com/download?id=1i8AJ7bGfrunJ_BA5_6G7-PX1CuRp3E66&confirm=t" | tar -xzf - -C data
# ETH ETF Floor F Jackal Data, split by north/south/central parts of the Hallway
curl -L "https://drive.usercontent.google.com/download?id=1cDQCbpdNvW4I7W7FuSAbRUjC6YXfC_CZ&confirm=t" | tar -xzf - -C data
# ETH ETZ Loading Dock Data
curl -L "https://drive.usercontent.google.com/download?id=1CXaehPlUjAxOP4HAczBgeNHD3FBdAG4w&confirm=t" | tar -xzf - -C data
# TUM RoboRacer/F1TENTH Data; Prepared and Split for Training, Eval and Testing
curl -L "https://drive.usercontent.google.com/download?id=1b9irVhwQQwqLVRQTrr_YfmU_B0Ko_ZWj&confirm=t" | tar -xzf - -C data

```
5. Run a script with `uv run python main.py`


## Recording data

1. Record data with ROS 2
```
ros2 bag record /livox/lidar /odom
```
2. Create a folder if not there yet and place ROSbag inside ~/Diff-PF/data

3. You can also create already a map 
```
cd ~/diff-slam-main
uv run python -m diff_slam.build_map --bag data/<ROSbag name> --output map.pt --lidar-topic /livox/lidar --odometry-topic /odom
```
## Train a model
Train the observation model on your dataset (`obs_sdf` for the SDF model, `obs_pp` for the pillar model). Datasets use the `data/<name>,<bag1>,<bag2>,...` format:
```
uv run python -m diff_slam.obs obs_sdf --train.train-ds "data/<dataset>,<bag1>,<bag2>"
```
The checkpoint is written to `models/<save-tag>/model.pt`.

To train a motion model on you dataset, similarly ues the following:
```
uv run python -m diff_slam.dynamics --train.train-ds "data/<dataset>,<bag1>,<bag2>" model:mlp-config
```

Train the end-to-end DPF, wiring in the pretrained observation model and a motion-model architecture (`odometry-se3`, `realnvp-se3`, `ensemble-se3`, or `mlp-se3`):
```
uv run python -m diff_slam.dpf --train.obs-model models/<save-tag>/model.pt --train.motion-arch mlp-se3 --train.train-ds "data/<dataset>,<bag1>"
```
You can specifiy the motion model checkpoint as follows:
```
uv run python -m diff_slam.dpf --train.obs-model models/<save-tag>/model.pt --train.motion models/<save-tag>/model.pt --train.train-ds "data/<dataset>,<bag1>"
```

The checkpoint is written to `models/dpf-<obs>-<motion>/model.pt`.

## Run live inference in ROS 2
1. Create a ros2_ws and link the existing folder to its src folder
```
mkdir -p ~/diffslam_ws/src
ln -s ~/diff-slam-main ~/diffslam_ws/src/diff_slam
```

2. Run the code in live inference mode
```
colcon build
source install/setup.bash
ros2 launch diff_slam localizer.launch.py map_path:=~/diff-slam-main/map.pt model_path:=~/diff-slam-main/models/<dpf-run>/model.pt pointcloud_topic:=/livox/lidar
```


## Evaluation

Run offline evaluation on a recorded bag with a specific model to generate and record evaluation metrics to csv, md and wandb.

```bash
uv run python -m diff_slam.eval --dataset data/<bag> --model models/<model>/model.pt --mode <mode>
```

Output is written to `eval_runs/metrics.csv`, `eval_runs/summary.md` and wandb.

Optionally, KITTI-style output paths can be written to `data/<bag>/eval/` with `--out`:
- `pred.txt` — estimated trajectory (KITTI format)
- `gt.txt` — ground truth trajectory (KITTI format)

**Ground truth sources**

| Source | When used | Notes |
|--------|-----------|-------|
| `pose_graph.g2o` | default | SLAM-derived pseudo-GT, produced by `build_map`. |
| `gt/` bag | `--mocap-gt` flag | Real mocap poses; requires a `gt/` sub-bag. Synced to lidar frames by bag recording timestamp. |
| On-the-fly SLAM | fallback | If no `pose_graph.g2o` is found, reference poses are recomputed via SLAM. |

**Flags**

| Flag | Description |
|------|-------------|
| `--dataset` | Path to the ROS2 bag directory (default: `data/jackal/flat`). |
| `--model` | Full self-contained DPF checkpoint (cannot combine with `--obs`/`--motion`). |
| `--obs` | Observation-model-only checkpoint; assemble a DPF around it. |
| `--motion` | Motion-model-only checkpoint (requires `--obs`). |
| `--mode` | Observation model type: `ppe`/`ppp` (pillar) or `sdf` (SDF-based). |
| `--mocap-gt` | Use mocap GT from `data/<bag>/gt/` instead of g2o pseudo-GT. |
| `--out` | Enable KITTI output and optionally specify path (default: `data/<bag>/eval/`). |
| `--results-dir`| Directory for metrics CSV and MD summary (default: `eval_runs/`). |
| `--downsample` | Point cloud downsample count (default: 8000). |
| `--steps` | Steps per eval sequence (default: 6). |
| `--no-log` | Disable CSV, MD, and WandB logging. |
| `--no-wandb` | Disable WandB logging. |
| `--lidar-topic`| ROS2 topic for PointCloud2 data (default: `/livox/lidar`). |
| `--odometry-topic`| ROS2 topic for Odometry data (default: `/odometry/filtered`). |


## Project structure
- `bag_parser.py` ROS2 bag parser to extract LIDAR & odometry data
  - `def extract_sync` extracts raw data and computes SLAM reference poses (disk-cached)
  - `def extract_mocap_gt` syncs mocap poses from a `gt/` sub-bag to lidar frame timestamps
  - `class LocalisationDataset` base pytorch `Dataset` wrapper for ROS2 recordings
- `build_map.py` builds a localization map from a ROS2 bag or PCD file; also saves `pose_graph.g2o`
- `eval.py` offline evaluation: runs DPF inference on a recorded bag and saves pred/GT trajectories in KITTI format
- `module.py` `ModuleConfig` / `ConfigRegistry`: per-model config dataclasses and checkpoint (config + weights) bundling
- `dpf.py` differentiable particle filter implementation & e2e training
  - `class SequenceDataset` dataset to derive short sequences for e2e training
  - `class DPFv2` differentiable particle filter implementation
  - `class DPFConfig` config bundling the dynamics, observation and resampling configs
  - `def train_dpf` training routine for the DPF (wires in a pretrained observation model)
- `obs.py` observation models & pre-training
  - `class AlignmentDataset` dataset with Gaussian-derived pseudo labels as alignment score
  - `class FixedPillarEncoder` point cloud voxel encoder with fixed feature extraction
  - `class StructuralComparator` pillar/voxel-based observation model
  - `class MultiSDF` / `class SDFObservationModel` neural SDF observation model
  - `def train_sdf`, `def main_sdf`, `def main_pp` observation-model training entry points
- `resample.py` differentiable resampling strategies (contraction, Gumbel-Softmax, optimal transport)
- `slam.py` reference point cloud registration using ICP 
- `util.py` helper functions
- `visualize.py` visualization helpers
- `/data` ROS2 bag storage
- `/cache` execution cache, generated automatically
- `/models` model checkpoints & training stats, generated automatically

## How to run

For all steps the training progress is displayed in the console and checkpoints will be written to `/models`. The necessary SLAM computations and bag extraction will happen automatically. The results are cached, deleting `/cache` and `/models` will force recomputation.

1. Pre-train the observation model, e.g. `uv run python -m diff_slam.obs obs_sdf --train.train-ds "data/<dataset>,<bag>"`; checkpoints are written to `models/<save-tag>/`.
2. Run e2e training for the entire DPF, e.g. `uv run python -m diff_slam.dpf --train.obs-model models/<save-tag>/model.pt --train.motion-arch mlp-se3`; checkpoints are written to `models/dpf-<obs>-<motion>/`.
