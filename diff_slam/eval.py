import argparse
import csv
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path
import secrets

import cpuinfo
import psutil
import torch
import wandb
import plotly.graph_objects as go
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy

from diff_slam.dynamics import DynamicsConfig, OdometrySE3Config
from diff_slam.resample import ContractResampleConfig
from diff_slam.bag_parser import compose_static_chain, extract_mocap_gt, LocalisationDataset, read_static_tf_graph
from diff_slam.module import ModuleConfig
from diff_slam.dpf import SequenceDataset, DPFv2, DPFConfig, init_se2_like_particles
from diff_slam.obs import obs_mode_label
from diff_slam.util import get_device, collate_fn, wandb_save_artifact
from diff_slam.pose import udq_mul, udq_conj, udq_from_qt, udq_to_matrix
from diff_slam import visualize as vis
from diff_slam.build_map import extract_sync

from evo.main_ape import ape
from evo.core import trajectory, metrics
import evo.cli.common_ape_rpe as common
from evo.tools.settings import SETTINGS


def build_model(
    dpf_checkpoint: Path | None = None,
    obs_checkpoint: Path | None = None,
    motion_checkpoint: Path | None = None,
    mode: str | None = None,
    device: torch.device = None,
) -> tuple[DPFv2, str]:
    if dpf_checkpoint and (obs_checkpoint or motion_checkpoint):
        raise ValueError("--dpf is self-contained; cannot combine with --obs/--motion")
    if motion_checkpoint and not obs_checkpoint:
        raise ValueError("--motion requires --obs")
    if not (dpf_checkpoint or obs_checkpoint):
        raise ValueError("provide --dpf or --obs")

    if dpf_checkpoint is not None:
        # Self-contained DPF bundle: config + weights for the whole filter.
        cfg, model = DPFConfig.load_module(dpf_checkpoint)
        resolved_mode = mode or obs_mode_label(cfg.observer)
    else:
        # Assemble a DPF from a pretrained observation model (+ optional motion).
        assert obs_checkpoint is not None
        obs_cfg, obs_model = ModuleConfig.load_module(obs_checkpoint)
        if motion_checkpoint is not None:
            _, motion_model = DynamicsConfig.load_module(motion_checkpoint)
        else:
            motion_model = OdometrySE3Config().build_model()
        model = DPFv2(
            dynamics=motion_model,
            observer=obs_model,
            resample=ContractResampleConfig().get_strategy(),
            particle_count=200,
            temp=1,
            init_noise=DPFConfig.init_noise, # TODO: implicitly using default, maybe it shouldn't
            odom_noise=DPFConfig.odom_noise,
        )
        resolved_mode = mode or obs_mode_label(obs_cfg)

    model.to(device).eval()
    return model, resolved_mode


def load_g2o(path: Path) -> torch.Tensor:
    """Parse VERTEX_SE3:QUAT entries from g2o file, return (N, 8) UDQ poses sorted by vertex ID."""
    vertices = {}
    with open(path) as f:
        for line in f:
            if not line.startswith('VERTEX_SE3:QUAT'):
                continue
            parts = line.split()
            vid = int(parts[1])
            t = torch.tensor([float(parts[2]), float(parts[3]), float(parts[4])])
            # g2o stores quat as (qx, qy, qz, qw); udq_from_qt expects (w, x, y, z)
            q = torch.tensor([float(parts[8]), float(parts[5]), float(parts[6]), float(parts[7])])
            vertices[vid] = udq_from_qt(q, t)
    return torch.stack([vertices[i] for i in sorted(vertices)])


def save_kitti(poses: torch.Tensor, path: Path) -> None:
    """Save (N, 4, 4) poses as KITTI odometry txt: one pose per line, 12 values (3x4 row-major)."""
    with open(path, 'w') as f:
        for pose in poses:
            values = pose[:3, :].flatten().tolist()
            f.write(' '.join(f'{v:.6e}' for v in values) + '\n')


def save_metrics(csv_path: Path, metrics_dict: dict, model: str, dataset: str, mode: str, run_id: str, plot_map: str, plot_raw: str, dist: float, latency: float, downsample: int, steps: int, hardware: str):
    """Append evaluation metrics to a CSV file, creating it with headers if it doesn't exist."""
    fieldnames = ['run_id', 'timestamp', 'model', 'dataset', 'mode', 'downsample', 'steps', 'step_size', 'hardware', 'dist', 'latency', 'rmse', 'mean', 'median', 'std', 'min', 'max', 'sse', 'plot_map', 'plot_raw']
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        row = {
            'run_id': run_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model,
            'dataset': dataset,
            'mode': mode,
            'downsample': downsample,
            'steps': steps,
            'step_size': 1,
            'hardware': hardware,
            'dist': dist,
            'latency': latency,
            'plot_map': plot_map,
            'plot_raw': plot_raw
        }
        
        # Add metrics, converting numpy types to standard floats
        for key in fieldnames[11:-2]:
            if key in metrics_dict:
                val = metrics_dict[key]
                row[key] = float(val) if hasattr(val, 'item') else val
            else:
                row[key] = None
                
        writer.writerow(row)


def update_markdown_report(csv_path: Path, md_path: Path):
    """Regenerate a markdown report table from the metrics CSV with a stacked layout."""
    if not csv_path.exists():
        return
        
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    if not rows:
        return
        
    with open(md_path, 'w') as f:
        f.write("# Evaluation Summary\n\n")
        f.write("| Run Info | Metrics | Map View | Raw View |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        for row in rows:
            # Format Run Info Block
            info = (
                f"**ID:** `{row.get('run_id', 'N/A')}`<br>"
                f"**Time:** {row.get('timestamp', 'N/A')}<br>"
                f"**Model:** {row.get('model', 'N/A')}<br>"
                f"**Data:** {row.get('dataset', 'N/A')}<br>"
                f"**Mode:** {row.get('mode', 'N/A')}<br>"
                f"**Distance:** {float(row.get('dist', 'N/A')):.2f}m<br>"
                f"**Params:** ds={row.get('downsample')}, st={row.get('steps')}, sz={row.get('step_size')}<br>"
                f"**Hardware:**<br>{row.get('hardware', 'N/A').replace(',','<br>')}"
            )
            
            # Format Metrics Block
            m_list = []
            for m in ['rmse', 'mean', 'median', 'std', 'sse']:
                if val := row.get(m):
                    try:
                        m_list.append(f"**{m.upper()}:** {float(val):.4f}")
                    except ValueError:
                        m_list.append(f"**{m.upper()}:** {val}")
            metrics_str = "<br>".join(m_list)
            metrics_str += f"<br>**LATENCY:** {float(row.get('latency')):.2f}ms"
            
            # Format Plot Blocks
            p_map = f"![map]({row['plot_map']})" if row.get('plot_map') else "N/A"
            p_raw = f"![raw]({row['plot_raw']})" if row.get('plot_raw') else "N/A"
            
            f.write(f"| {info} | {metrics_str} | {p_map} | {p_raw} |\n")


def log_interactive_trajectory(traj_gt: torch.Tensor, traj_pred: torch.Tensor, world_np: numpy.ndarray = None, surface_points_np: numpy.ndarray = None, name: str = "Trajectory Comparison"):
    """Create a Plotly 3D scatter plot for trajectories and log to WandB."""
    fig = go.Figure()
    
    t_gt = traj_gt[:, :3, 3]
    t_pred = traj_pred[:, :3, 3]
    
    fig.add_trace(go.Scatter3d(
        x=t_gt[:, 0], y=t_gt[:, 1], z=t_gt[:, 2],
        mode='lines',
        line=dict(color='blue', width=4),
        name='Ground Truth'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=t_pred[:, 0], y=t_pred[:, 1], z=t_pred[:, 2],
        mode='lines',
        line=dict(color='red', width=4),
        name='Prediction'
    ))

    # Add Map if provided
    if world_np is not None:
        fig.add_trace(go.Scatter3d(
            x=world_np[:, 0], y=world_np[:, 1], z=world_np[:, 2],
            mode='markers',
            marker=dict(size=1, color=world_np[:, 2], colorscale='Viridis', opacity=0.3),
            name='World Map'
        ))

    # Add SDF Surface if provided
    if surface_points_np is not None:
        fig.add_trace(go.Scatter3d(
            x=surface_points_np[:, 0], y=surface_points_np[:, 1], z=surface_points_np[:, 2],
            mode='markers',
            marker=dict(size=2, color='orange', opacity=0.6),
            name='SDF Surface'
        ))

    fig.update_layout(
        title=name,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        margin=dict(r=0, l=0, b=0, t=30)
    )
    
    wandb.log({"eval/interactive_trajectory": wandb.Plotly(fig)})


def run_eval(
    data_path: Path,
    dpf_checkpoint: Path | None,
    obs_checkpoint: Path | None,
    motion_checkpoint: Path | None,
    lidar_topic: str,
    odometry_topic: str,
    res_dir: Path,
    output_dir: Path | None = None,
    downsample: int = None,
    steps: int = None,
    mocap_gt: Path = None,
    mode: str = "ppe",
    log: bool = True,
    use_wandb: bool = True,
    wandb_entity: str | None = None,
    wandb_project: str = "diff-slam",
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    show_map: bool = True,
    log_3d: bool = False,
    trim_ends: bool = False,
):
    device = get_device()
    model, mode = build_model(
        dpf_checkpoint=dpf_checkpoint,
        obs_checkpoint=obs_checkpoint,
        motion_checkpoint=motion_checkpoint,
        mode=mode,
        device=device,
    )
    checkpoint = dpf_checkpoint or obs_checkpoint

    # Potentially Overwrite Model Params
    if int(model.downsample) > 0:
        if downsample and downsample != int(model.downsample):
            print(f"Overriding downsample {int(model.downsample)} from model --> {downsample}")
            model.register_buffer('downsample', torch.tensor(downsample))
        else:
            downsample = int(model.downsample)
    else:
        if downsample is None:
            raise ValueError("Must specify downsample via argument or model checkpoint")
    if int(model.steps) > 0:
        if steps and steps != int(model.steps):
            print(f"Overriding steps {int(model.steps)} from model -> {steps}")
            model.register_buffer('steps', torch.tensor(steps))
        else:
            steps = int(model.steps)
    else:
        if steps is None:
            raise ValueError("Must specify steps via argument or model checkpoint")
        else:
            model.register_buffer('steps', torch.tensor(steps))
    
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-{secrets.token_hex(4)}"

    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else "None"
    cpu_name = cpuinfo.get_cpu_info().get('brand_raw', "Unknown CPU")
    ram_gb = psutil.virtual_memory().total / (1024**3)
    hardware_str = f"{cpu_name} , {gpu_name} , {ram_gb:.1f}GB RAM"

    base_ds = LocalisationDataset(data_path, odometry_topic=odometry_topic)
    if trim_ends:
        base_ds.scans = base_ds.scans[1:-1]
        base_ds.poses = base_ds.poses[1:-1]
        base_ds.odometry = base_ds.odometry[1:-1]
        print(f'trimmed first/last dataset frame: {len(base_ds.scans)} frames remain')
    
    ds = SequenceDataset(base_ds, step_count=2, downsample=downsample)
    dl = DataLoader(ds, shuffle=False, collate_fn=collate_fn)

    world_device = base_ds.world.to(device)

    world_np, sdf_np = None, None
    if show_map or log_3d:
        world_np = vis.get_world_map_3d(base_ds)
        if show_map:
            vis.show_world_np(world_np)
        if mode == "sdf":
            sdf_np = vis.get_sdf_map_3d(model.observer.sdf)
            if show_map:
                vis.show_sdf_np(sdf_np)

    g2o_path = None
    if mocap_gt is not None:
        if mocap_gt == Path("auto"):
            mocap_gt = data_path / 'gt'
        if mocap_gt.exists():
            # Get base -> lidar from static TF / KNOWN_STATIC_TF
            graph = read_static_tf_graph(data_path)
            # Find any topic to get the lidar frame id
            from rosbags.highlevel import AnyReader
            from rosbags.typesys import get_typestore, Stores
            with AnyReader([data_path], default_typestore=get_typestore(Stores.ROS2_KILTED)) as reader:
                lidar_conn = next(c for c in reader.connections if c.topic == lidar_topic)
                for _, _, raw in reader.messages(connections=[lidar_conn]):
                    msg = reader.deserialize(raw, lidar_conn.msgtype)
                    lidar_frame = msg.header.frame_id.lstrip('/')
                    break
            
            # Use 'base_link' as the assumed mocap target frame
            base_to_lidar = compose_static_chain(graph, 'base_link', lidar_frame)
            
            g2o_sub = extract_mocap_gt(data_path, mocap_gt, lidar_topic=lidar_topic, base_to_lidar=base_to_lidar, swap_forward=True)
            print(f'loaded mocap GT: {mocap_gt}  ({len(g2o_sub)} poses)')
        else:
            raise FileNotFoundError(f"Cannot find mocap bag \"{mocap_gt}\"")
    else:
        g2o_path = data_path / 'pose_graph.g2o'
        g2o_sub = None
        if not g2o_path.exists():
            print('gt g2o unavailable, generating poses via SLAM from bag...')
            _, poses_udq, _ = extract_sync(data_path, lidar_topic, odometry_topic)
            g2o_sub = poses_udq
        else:
            g2o_sub = load_g2o(g2o_path)
            print(f'loaded g2o GT: {g2o_path}  ({len(g2o_sub)} vertices)')

    if use_wandb:
        train_run = {}
        wandb_run_json = checkpoint.parent / "wandb_run.json"
        if wandb_run_json.exists():
            with open(wandb_run_json) as f:
                train_run = json.load(f)

        wandb.init(
            dir="eval_runs",
            entity=wandb_entity or train_run.get("entity") or None,
            project=wandb_project or train_run.get("project", "diff-pf-eval"),
            name=wandb_name or f"eval_{run_id}",
            group=wandb_group or train_run.get("group") or checkpoint.parent.name,
            config={
                "model_path": str(checkpoint) if checkpoint else "None",
                "obs_model_path": str(obs_checkpoint) if obs_checkpoint else "None",
                "dataset": str(data_path),
                "mode": mode,
                "downsample": downsample,
                "steps": steps,
                "step_size": 1,
                "particles": int(model.particle_count),
                "temp": float(model.temp),
                "resample": getattr(model.resample, 'func', model.resample).__name__,
                "mocap_gt": str(mocap_gt) if mocap_gt else "None",
                "g2o_gt": str(g2o_path) if g2o_path and g2o_path.exists() else "None",
                "hardware": hardware_str,
                "hostname": socket.gethostname(),
            }
        )
        wandb_save_artifact(str(checkpoint), f"eval-{run_id}-model", "model")


    traj_pred = []
    traj_gt = []
    inf_times = []
    # Use CUDA events for high-precision, non-blocking timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dl, desc='eval')):
            # Optimization: Only transfer necessary data to GPU. The world map is already on device.
            _, scans_cpu, _, odometry_cpu = batch
            scans = scans_cpu.to(device)
            odometry = odometry_cpu.to(device)

            if i == 0:
                x0 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device=device)
                p0 = udq_mul(init_se2_like_particles(model.particle_count, model.init_noise[0], model.init_noise[1]).to(device), x0)
                # x0 = odometry[0]
            else:
                x0 = traj_pred[i-1].to(device)
                p0 = model.p
                
            u = udq_mul(udq_conj(odometry[:-1]), odometry[1:])

            if device.type == 'cuda':
                start_event.record()
                x_out = model(scans, x0, p0, u, world_device)
                end_event.record()
                end_event.synchronize()
                inf_times.append(start_event.elapsed_time(end_event) / 1000)
            else:
                t0 = time.time()
                x_out = model(scans, x0, p0, u, world_device)
                inf_times.append(time.time() - t0)
            
            if use_wandb:
                wandb.log({"inference/latency": inf_times[-1] * 1000}, step=i)

            traj_pred.append(x_out[0].cpu())
            traj_gt.append(g2o_sub[i+1].cpu())

    avg_latency = (sum(inf_times[1:]) / len(inf_times[1:])) * 1000

    traj_gt = udq_to_matrix(torch.stack(traj_gt))
    traj_pred = udq_to_matrix(torch.stack(traj_pred))

    xyz_gt = traj_gt[:, :3, 3]
    diffs = xyz_gt[1:] - xyz_gt[:-1]
    total_dist = float(torch.norm(diffs, dim=1).sum())

    if output_dir is not None:
        if output_dir == Path("auto"):
            output_dir = data_path / "eval"
            
        os.makedirs(output_dir, exist_ok=True)

        pred_path = output_dir / 'pred.txt'
        save_kitti(traj_pred, pred_path)
        print(f'saved → {pred_path}')

        gt_path = output_dir / 'gt.txt' if not mocap_gt else output_dir / 'gt_mocap.txt'
        save_kitti(traj_gt, gt_path)
        print(f'saved → {gt_path}')

    traj_ref=trajectory.PosePath3D(poses_se3=traj_gt.numpy())
    traj_est=trajectory.PosePath3D(poses_se3=traj_pred.numpy())

    res = ape(
        traj_ref=traj_ref,
        traj_est=traj_est, 
        pose_relation=metrics.PoseRelation.translation_part,
        align=True,
    )

    print(res.pretty_str(True, True, False))

    if log:
        plots_dir = res_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

    common.plot_result(
        argparse.Namespace(
            plot_mode=SETTINGS.plot_mode_default,
            plot_x_dimension = "seconds",
            plot_colormap_min = None,
            plot_colormap_max = None,
            plot_colormap_max_percentile = None,
            map_tile = False,
            ros_map_yaml = False,
            plot=not log,
            save_plot=str(plots_dir / run_id) if log else None,
            no_warnings=False,
        ),
        res,
        traj_ref,
        traj_est
    )
    if log:
        vis.show_score(data_path, model.observer, save_filepath=f"{plots_dir / run_id}_score.png")

    if log:
        csv_path = res_dir / "metrics.csv"
        save_metrics(
            csv_path,
            res.stats,
            model=str(checkpoint),
            dataset=str(data_path),
            mode=mode,
            run_id=run_id,
            plot_map=f"plots/{run_id}_map.png",
            plot_raw=f"plots/{run_id}_raw.png",
            dist=total_dist,
            latency=avg_latency,
            downsample=downsample,
            steps=model.steps,
            hardware=hardware_str
        )
        
        update_markdown_report(csv_path, res_dir / "summary.md")

    if use_wandb:
        wandb.log({
            "eval/plots/map": wandb.Image(str(plots_dir / f"{run_id}_map.png")),
            "eval/plots/raw": wandb.Image(str(plots_dir / f"{run_id}_raw.png")),
            "eval/plots/obs_score": wandb.Image(str(plots_dir / f"{run_id}_score.png"))
        })

        metrics_to_log = {f"eval/{k}": v for k, v in res.stats.items()}
        metrics_to_log["eval/avg_latency"] = avg_latency
        metrics_to_log["eval/total_distance"] = total_dist

        for k, v in metrics_to_log.items():
            wandb.summary[k] = v

        log_interactive_trajectory(
            numpy.stack(traj_ref.poses_se3),
            numpy.stack(traj_est.poses_se3),
            world_np=world_np if log_3d else None,
            surface_points_np=sdf_np if log_3d else None,
            name=f"Trajectory: {run_id}"
        )

        wandb.finish()

    return

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate trained DPF model')

    parser.add_argument('--dataset', type=Path, default=Path('data/ethetf-floorf_26-05-10/etf_floorF/init'), help="Test Dataset")
    parser.add_argument('--model', type=Path, default=None, help="Full DPF checkpoint (self-contained, cannot combine with --obs/--motion)")
    parser.add_argument('--obs', type=Path, default=None, help="Observation model-only checkpoint")
    parser.add_argument('--motion', type=Path, default=None, help="Motion model-only checkpoint (requires --obs)")
    parser.add_argument('--mode', default=None, help='OBS System type', choices=["ppe", "sdf", "ppp"])
    parser.add_argument('--results-dir', type=Path, default='eval_runs/', help="Directory for evaluation results")

    parser.add_argument('--lidar-topic',    default='/livox/lidar',        help='(g2o-less only) Rosbag PointCloud2 topic')
    parser.add_argument('--odometry-topic', default='/odometry/filtered',  help='(g2o-less only) Rosbag Odometry topic')

    parser.add_argument('--downsample', type=int)
    parser.add_argument('--steps', type=int)

    parser.add_argument('--out', nargs='?', const="auto", type=Path, default=None, help="Output for kitti-type gt/pred files")
    parser.add_argument('--mocap-gt', nargs='?', const="auto", type=Path, default=None, help='Use mocap poses from gt/bag path instead of g2o pseudo-GT')
    parser.add_argument('--show-maps', action='store_true', help="SDF/World map visualization")

    parser.add_argument('--no-log', action='store_true', help="Do not add eval run to csv file, dont update md file and dont write wandb")
    parser.add_argument('--no-csv', action='store_true', help="Do not add eval run to csv file or md file")
    
    parser.add_argument('--no-wandb', action='store_true', help="Disable WandB logging")
    parser.add_argument('--wandb-entity', default="ethz-pbl", help="WandB entity (team or user)")
    parser.add_argument('--wandb-project', default="diff-pf-eval", help="WandB project name")
    parser.add_argument('--wandb-name', default=None, help="WandB run name, defaults to eval_UID")
    parser.add_argument('--wandb-group', default=None, help="WandB group name, defaults to the directory name the model is in")
    parser.add_argument('--wandb-no-log-3d', action='store_true', help="Log 3D map/SDF to WandB")

    parser.add_argument('--trim-ends', action='store_true', help='Drop first and last dataset frame before evaluation')

    return parser

def main():
    args = build_parser().parse_args()

    run_eval(
        data_path=args.dataset,
        dpf_checkpoint=args.model,
        obs_checkpoint=args.obs,
        motion_checkpoint=args.motion,
        lidar_topic=args.lidar_topic,
        odometry_topic=args.odometry_topic,
        res_dir=args.results_dir,
        output_dir=args.out,
        downsample=args.downsample,
        steps=args.steps,
        mocap_gt=args.mocap_gt,
        mode=args.mode,
        log=False if (args.no_log or args.no_csv) else True,
        use_wandb=False if (args.no_log or args.no_wandb) else True,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_group=args.wandb_group,
        show_map=args.show_maps,
        log_3d=not args.wandb_no_log_3d,
        trim_ends=args.trim_ends,
    )

if __name__ == '__main__':
    main()
