"""HGGD grasp-detection server for the ROS2 bridge.

This is the *worker* half of the two-process design used by the
``hggd_grasp_inference`` ROS2 package:

    [ this worker, conda env: torch+CUDA ]  --TCP/JSON-->  [ rclpy node, py3.12 ]

It owns the Intel RealSense D435 (via pyrealsense2) and the HGGD model
(AnchorGraspNet -> PointMultiGraspNet), exactly like ``demo_realsense.py``,
but instead of opening an Open3D window it streams the detected 6-DoF grasps
as newline-delimited JSON over a local TCP socket. The ROS2 node connects as a
client, converts each grasp to a geometry_msgs/Pose, and publishes them.

Run it from the HGGD repo root (so ``models``/``dataset``/``customgraspnetAPI``
import correctly), in the conda env that has torch+CUDA+pytorch3d+pyrealsense2:

    conda activate hggd-cu128            # the env with torch+cuda
    cd /home/thuongpc/code/HGGD
    python hggd_grasp_server.py --checkpoint-path ./HGGD_realsense_checkpoint

Defaults below match demo_realsense.sh, so no extra flags are required.
"""
import argparse
import json
import random
import socket
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import pyrealsense2 as rs

from dataset.evaluation import (anchor_output_process, collision_detect,
                                detect_2d_grasp, detect_6d_grasp_multi)
from dataset.pc_dataset_tools import data_process, feature_fusion
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet
from train_utils import *  # noqa: F401,F403  (matches demo_realsense.py)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', default='./HGGD_realsense_checkpoint')

# bridge socket
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', type=int, default=6000)
parser.add_argument('--frame-id', default='camera_color_optical_frame',
                    help='frame the grasp poses are expressed in (ROS optical)')

# realsense stream config
parser.add_argument('--rs-width', type=int, default=1280)
parser.add_argument('--rs-height', type=int, default=720)
parser.add_argument('--rs-fps', type=int, default=30)

# 2d (defaults mirror demo_realsense.sh)
parser.add_argument('--input-h', type=int, default=360)
parser.add_argument('--input-w', type=int, default=640)
parser.add_argument('--sigma', type=int, default=10)
parser.add_argument('--use-depth', type=int, default=1)
parser.add_argument('--use-rgb', type=int, default=1)
parser.add_argument('--ratio', type=int, default=8)
parser.add_argument('--anchor-k', type=int, default=6)
parser.add_argument('--anchor-w', type=float, default=50.0)
parser.add_argument('--anchor-z', type=float, default=20.0)
parser.add_argument('--grid-size', type=int, default=8)

# pc
parser.add_argument('--anchor-num', type=int, default=7)
parser.add_argument('--all-points-num', type=int, default=25600)
parser.add_argument('--center-num', type=int, default=48)
parser.add_argument('--group-num', type=int, default=512)

# grasp detection
parser.add_argument('--heatmap-thres', type=float, default=0.01)
parser.add_argument('--local-k', type=int, default=10)
parser.add_argument('--local-thres', type=float, default=0.01)
parser.add_argument('--rotation-num', type=int, default=1)

# others
parser.add_argument('--random-seed', type=int, default=123)
parser.add_argument('--max-depth-mm', type=float, default=1000.0,
                    help='depth clip in mm (model trained with 1000mm)')

args = parser.parse_args()


class PointCloudHelper:
    """Back-projection maps built from the live camera intrinsics.

    Identical to demo_realsense.py (arrays indexed [W, H] to match the
    transposed layout used throughout the HGGD demo code).
    """

    def __init__(self, all_points_num, fx, fy, cx, cy, width, height) -> None:
        self.all_points_num = all_points_num
        self.output_shape = (80, 45)
        self.width, self.height = width, height
        ymap, xmap = np.meshgrid(np.arange(height), np.arange(width))
        points_x = (xmap - cx) / fx
        points_y = (ymap - cy) / fy
        self.points_x = torch.from_numpy(points_x).float()
        self.points_y = torch.from_numpy(points_y).float()
        ymap, xmap = np.meshgrid(np.arange(self.output_shape[1]),
                                 np.arange(self.output_shape[0]))
        factor = width / self.output_shape[0]
        points_x = (xmap - cx / factor) / (fx / factor)
        points_y = (ymap - cy / factor) / (fy / factor)
        self.points_x_downscale = torch.from_numpy(points_x).float()
        self.points_y_downscale = torch.from_numpy(points_y).float()

    def to_scene_points(self, rgbs, depths, include_rgb=True):
        batch_size = rgbs.shape[0]
        feature_len = 3 + 3 * include_rgb
        points_all = -torch.ones(
            (batch_size, self.all_points_num, feature_len),
            dtype=torch.float32).cuda()
        idxs = []
        masks = (depths > 0)
        cur_zs = depths / 1000.0
        cur_xs = self.points_x.cuda() * cur_zs
        cur_ys = self.points_y.cuda() * cur_zs
        for i in range(batch_size):
            points = torch.stack([cur_xs[i], cur_ys[i], cur_zs[i]], axis=-1)
            mask = masks[i]
            points = points[mask]
            colors = rgbs[i][:, mask].T
            if len(points) >= self.all_points_num:
                cur_idxs = random.sample(range(len(points)),
                                         self.all_points_num)
                points = points[cur_idxs]
                colors = colors[cur_idxs]
                idxs.append(cur_idxs)
            if include_rgb:
                points_all[i] = torch.concat([points, colors], axis=1)
            else:
                points_all[i] = points
        return points_all, idxs, masks

    def to_xyz_maps(self, depths):
        downsample_depths = F.interpolate(depths[:, None],
                                          size=self.output_shape,
                                          mode='nearest').squeeze(1).cuda()
        cur_zs = downsample_depths / 1000.0
        cur_xs = self.points_x_downscale.cuda() * cur_zs
        cur_ys = self.points_y_downscale.cuda() * cur_zs
        xyzs = torch.stack([cur_xs, cur_ys, cur_zs], axis=-1)
        return xyzs.permute(0, 3, 1, 2)


def patch_intrinsics(fx, fy, cx, cy):
    """Make HGGD back-project grasp translations with the *live* D435
    intrinsics instead of the hardcoded matrix in dataset/config.py.

    grasp.py does ``from .config import get_camera_intrinsic`` and calls it at
    runtime inside to_6d_grasp_group(), so we override both the source binding
    and the one already imported into grasp.py's namespace.
    """
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def _live_intrinsic(camera=None):
        return K

    import dataset.config as _cfg
    import dataset.grasp as _grasp
    _cfg.get_camera_intrinsic = _live_intrinsic
    _grasp.get_camera_intrinsic = _live_intrinsic
    print('-> patched grasp intrinsics with live D435 matrix')


def preprocess(pc_helper, rgb_np, depth_np):
    """rgb_np: HxWx3 uint8 (RGB). depth_np: HxW float (millimeters)."""
    ori_rgb = rgb_np / 255.0
    ori_depth = np.clip(depth_np, 0, args.max_depth_mm)
    ori_rgb = torch.from_numpy(ori_rgb).permute(2, 1, 0)[None]
    ori_rgb = ori_rgb.to(device='cuda', dtype=torch.float32)
    ori_depth = torch.from_numpy(ori_depth.astype(np.float32)).T[None]
    ori_depth = ori_depth.to(device='cuda', dtype=torch.float32)

    view_points, _, _ = pc_helper.to_scene_points(ori_rgb, ori_depth,
                                                  include_rgb=True)
    xyzs = pc_helper.to_xyz_maps(ori_depth)

    rgb = F.interpolate(ori_rgb, (args.input_w, args.input_h))
    depth = F.interpolate(ori_depth[None], (args.input_w, args.input_h))[0]
    depth = depth / 1000.0
    depth = torch.clip((depth - depth.mean()), -1, 1)
    x = torch.concat([depth[None], rgb], 1).to(device='cuda',
                                               dtype=torch.float32)
    return view_points, xyzs, x, ori_depth


def inference(anchornet, localnet, anchors, view_points, xyzs, x, ori_depth):
    with torch.no_grad():
        pred_2d, perpoint_features = anchornet(x)
        loc_map, cls_mask, theta_offset, height_offset, width_offset = \
            anchor_output_process(*pred_2d, sigma=args.sigma)
        rect_gg = detect_2d_grasp(loc_map, cls_mask, theta_offset,
                                  height_offset, width_offset,
                                  ratio=args.ratio, anchor_k=args.anchor_k,
                                  anchor_w=args.anchor_w, anchor_z=args.anchor_z,
                                  mask_thre=args.heatmap_thres,
                                  center_num=args.center_num,
                                  grid_size=args.grid_size,
                                  grasp_nms=args.grid_size, reduce='max')
        if rect_gg.size == 0:
            return None

        points_all = feature_fusion(view_points[..., :3], perpoint_features,
                                    xyzs)
        rect_ggs = [rect_gg]
        pc_group, valid_local_centers = data_process(
            points_all, ori_depth, rect_ggs, args.center_num, args.group_num,
            (args.input_w, args.input_h), min_points=32, is_training=False)
        rect_gg = rect_ggs[0]
        points_all = points_all.squeeze()

        grasp_info = np.zeros((0, 3), dtype=np.float32)
        g_thetas = rect_gg.thetas[None]
        g_ws = rect_gg.widths[None]
        g_ds = rect_gg.depths[None]
        cur_info = np.vstack([g_thetas, g_ws, g_ds])
        grasp_info = np.vstack([grasp_info, cur_info.T])
        grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32,
                                                     device='cuda')

        _, pred, offset = localnet(pc_group, grasp_info)
        _, pred_rect_gg = detect_6d_grasp_multi(rect_gg, pred, offset,
                                                valid_local_centers,
                                                (args.input_w, args.input_h),
                                                anchors, k=args.local_k)
        pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group(depth=0.02)
        pred_gg, _ = collision_detect(points_all, pred_grasp_from_rect,
                                      mode='graspnet')
        pred_gg = pred_gg.nms()
        return pred_gg


def grasps_to_payload(pred_gg, frame_index):
    """Serialize a 6-DoF GraspGroup into a JSON-able dict, sorted best-first."""
    grasps = []
    if pred_gg is not None and len(pred_gg) > 0:
        order = np.argsort(-np.asarray(pred_gg.scores).reshape(-1))
        for i in order:
            t = np.asarray(pred_gg.translations[i], dtype=float).reshape(3)
            R = np.asarray(pred_gg.rotations[i], dtype=float).reshape(3, 3)
            grasps.append({
                't': t.tolist(),
                'R': R.reshape(9).tolist(),
                'width': float(pred_gg.widths[i]),
                'depth': float(pred_gg.depths[i]),
                'score': float(pred_gg.scores[i]),
            })
    return {
        'frame': int(frame_index),
        'frame_id': args.frame_id,
        'grasps': grasps,
    }


def main():
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # --- start RealSense pipeline (depth aligned to color) ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.rs_width, args.rs_height,
                         rs.format.rgb8, args.rs_fps)
    config.enable_stream(rs.stream.depth, args.rs_width, args.rs_height,
                         rs.format.z16, args.rs_fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_intr = profile.get_stream(rs.stream.color) \
        .as_video_stream_profile().get_intrinsics()
    fx, fy = color_intr.fx, color_intr.fy
    cx, cy = color_intr.ppx, color_intr.ppy
    print(f'-> D435 intrinsics fx={fx:.2f} fy={fy:.2f} '
          f'cx={cx:.2f} cy={cy:.2f}, depth_scale={depth_scale}')

    patch_intrinsics(fx, fy, cx, cy)
    pc_helper = PointCloudHelper(args.all_points_num, fx, fy, cx, cy,
                                 args.rs_width, args.rs_height)

    # --- model ---
    anchornet = AnchorGraspNet(in_dim=4, ratio=args.ratio,
                               anchor_k=args.anchor_k).cuda()
    localnet = PointMultiGraspNet(info_size=3, k_cls=args.anchor_num**2).cuda()
    check_point = torch.load(args.checkpoint_path)
    anchornet.load_state_dict(check_point['anchor'])
    localnet.load_state_dict(check_point['local'])
    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1).cuda()
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
    anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
    anchors['gamma'] = check_point['gamma']
    anchors['beta'] = check_point['beta']
    anchornet.eval()
    localnet.eval()
    print('-> loaded checkpoint %s' % args.checkpoint_path)

    # warm up / let auto-exposure settle
    for _ in range(15):
        pipeline.wait_for_frames()

    # --- TCP server: one ROS client at a time ---
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f'-> grasp server listening on {args.host}:{args.port} '
          f'(frame_id={args.frame_id}). Waiting for ROS node...')

    frame_index = 0
    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f'-> ROS node connected from {addr}')
            try:
                while True:
                    frames = align.process(pipeline.wait_for_frames())
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                    rgb_np = np.asanyarray(color_frame.get_data())
                    depth_np = np.asanyarray(depth_frame.get_data()).astype(
                        np.float32)
                    depth_np = depth_np * depth_scale * 1000.0

                    view_points, xyzs, x, ori_depth = preprocess(
                        pc_helper, rgb_np, depth_np)
                    t0 = time()
                    pred_gg = inference(anchornet, localnet, anchors,
                                        view_points, xyzs, x, ori_depth)
                    dt = (time() - t0) * 1e3

                    payload = grasps_to_payload(pred_gg, frame_index)
                    line = (json.dumps(payload) + '\n').encode('utf-8')
                    conn.sendall(line)
                    print('frame %d: %d grasps, %.1f ms' %
                          (frame_index, len(payload['grasps']), dt))
                    frame_index += 1
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                print(f'-> ROS node disconnected ({e}); waiting for reconnect')
            finally:
                conn.close()
    except KeyboardInterrupt:
        print('\n-> stopping')
    finally:
        srv.close()
        pipeline.stop()


if __name__ == '__main__':
    main()
