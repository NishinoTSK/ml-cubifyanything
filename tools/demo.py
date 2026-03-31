# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import glob
import itertools
import json
import numpy as np
import rerun
import rerun.blueprint as rrb
import torch
import torchvision
import sys
import uuid

from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

from cubifyanything.batching import Sensors
from cubifyanything.boxes import GeneralInstance3DBoxes
from cubifyanything.capture_stream import CaptureDataset
from cubifyanything.color import random_color
from cubifyanything.cubify_transformer import make_cubify_transformer
from cubifyanything.dataset import CubifyAnythingDataset
from cubifyanything.instances import Instances3D
from cubifyanything.preprocessor import Augmentor, Preprocessor

def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    try:
        return src.to(dst)
    except:
        return src.to(dst.device)

def move_to_current_device(x, t):
    if isinstance(x, (list, tuple)):
        return [move_device_like(x_, t) for x_ in x]
    
    return move_device_like(x, t)

def move_input_to_current_device(batched_input: Sensors, t: torch.Tensor):
    # Assume only two levels of nesting for now.
    return { name: { name_: move_to_current_device(m, t) for name_, m in s.items() } for name, s in batched_input.items() }

# A global dictionary we use to for consistent colors for instances across frames.
ID_TO_COLOR = {}

def log_instances(instances, prefix, boxes_3d_name="gt_boxes_3d", ids_name="gt_ids", captions_name="gt_captions", log_instances_name="instances", **kwargs):
    global ID_TO_COLOR
    boxes_3d = instances.get(boxes_3d_name)

    colors = []    
    if instances.has(ids_name):
        ids = instances.get(ids_name)
        for id_ in ids:
            ID_TO_COLOR[id_] = ID_TO_COLOR.get(id_, random_color(rgb=True))
            colors.append(ID_TO_COLOR[id_])
    else:
        ids = None
        colors = [random_color(rgb=True) for _ in range(len(instances))]

    captions = None
    if instances.has(captions_name):
        captions = instances.get(captions_name)

    # Make captions prominent, but include IDs if present.
    if captions is not None:
        labels = captions
    else:
        labels = len(instances) * [""]

    if ids is not None:
        labels = [f"{label} ({id})" if label else str(id) for label, id in zip(labels, ids)]

    quaternions = [
        rerun.Quaternion(
            xyzw=Rotation.from_matrix(r).as_quat()
        )

        for r in boxes_3d.R.cpu().numpy()
    ]

    # Hard-code these suffixes.
    rerun.log(
        f"{prefix}/{log_instances_name}",
        rerun.Boxes3D(
            centers=boxes_3d.gravity_center.cpu().numpy(),
            sizes=boxes_3d.dims.cpu().numpy(),
            quaternions=quaternions,
            colors=colors,
            labels=labels,
            show_labels=True),
        **kwargs)

def load_data_and_visualize(dataset):
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            contents=[
                rrb.Spatial3DView(
                    name="World",
                    origin="/world"),
                rrb.Horizontal(
                    contents=[
                        rrb.Spatial2DView(
                            name="Image",
                            origin="/device/wide/image",
                            contents=[
                                "+ $origin/**",
                                "+ /device/wide/instances/**"
                            ]),
                        rrb.Spatial2DView(
                            name="Depth",
                            origin="/device/wide/depth"),
                        rrb.Spatial2DView(
                            name="Depth (GT)",
                            origin="/device/gt/depth"),
                    ],
                    name="Wide")
            ]))

    recording = None
    video_id = None
    for sample in dataset:
        sample_video_id = sample["meta"]["video_id"]
        if (recording is None) or (video_id != sample_video_id):
            new_recording = rerun.new_recording(
                application_id=str(sample_video_id), recording_id=uuid.uuid4(), make_default=True)            

            new_recording.send_blueprint(blueprint, make_active=True)
            rerun.spawn()

            recording = new_recording
            video_id = sample_video_id
        
        # Check for the world. Note that this may not show if --every-nth-frame is used.
        if "world" in sample:
            world_instances = sample["world"]["instances"]
            log_instances(world_instances, prefix="/world", static=True)
            continue

        rerun.set_time_seconds("pts", sample["meta"]["timestamp"], recording=recording)

        # -> channels last.
        image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)        
        camera = rerun.Pinhole(
            image_from_camera=sample["sensor_info"].wide.image.K[-1].numpy(), resolution=sample["sensor_info"].wide.image.size)

        # Log this to both the device (per-frame) and to the world.
        rerun.log("/device/wide/image", rerun.Image(image).compress())
        rerun.log("/device/wide/image", camera)

        # RT here corresponds to the laser-scanner space, as registered to the capture device, so this allows us
        # to visualize the camera with respect to the annotation space.
        RT = sample["sensor_info"].gt.RT[-1].numpy()
        pose_transform = rerun.Transform3D(
            translation=RT[:3, 3],
            rotation=rerun.Quaternion(xyzw=Rotation.from_matrix(RT[:3, :3]).as_quat()))
        
        rerun.log("/world/image", pose_transform)
        rerun.log("/world/image", camera)
        rerun.log("/world/image/image", rerun.Image(image, opacity=0.5))

        rerun.log("/device/wide/depth", rerun.DepthImage(sample["wide"]["depth"][-1].numpy()))
        rerun.log("/device/gt/depth", rerun.DepthImage(sample["gt"]["depth"][-1].numpy()))

        per_frame_instances = sample["wide"]["instances"]
        log_instances(per_frame_instances, prefix="/device/wide")

def get_camera_coords(depth):
    height, width = depth.shape
    device = depth.device

    # camera xy.
    camera_coords = torch.stack(
        torch.meshgrid(
            torch.arange(0, width, device=device),
            torch.arange(0, height, device=device), indexing="xy"),
        dim=-1)

    return camera_coords

def unproject(depth, K, RT, max_depth=10.0):
    camera_coords = get_camera_coords(depth) * depth[..., None]

    intrinsics_4x4 = torch.eye(4, device=depth.device)
    intrinsics_4x4[:3, :3] = K

    valid = depth > 0
    if max_depth is not None:
        valid &= (depth < max_depth)

    depth = depth[..., None]
    uvd = torch.cat((camera_coords, depth, torch.ones_like(depth)), dim=-1)

    camera_xyz =  torch.linalg.inv(intrinsics_4x4) @ uvd.view(-1, 4).T
    world_xyz = RT @ camera_xyz

    return world_xyz.T[..., :-1].reshape(uvd.shape[0], uvd.shape[1], 3), valid

def _to_builtin(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def _safe_frame_stem(video_id, timestamp_seconds):
    # Keep filenames stable and Windows-friendly.
    return f"{int(video_id)}_{timestamp_seconds:.6f}".replace(".", "p")

def save_pred_bboxes_json(pred_instances: Instances3D, sample_meta: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    video_id = sample_meta.get("video_id")
    timestamp = sample_meta.get("timestamp")
    image_h, image_w = pred_instances.image_size

    dets = []
    n = len(pred_instances) if pred_instances is not None else 0
    for i in range(n):
        det = {}
        if pred_instances.has("pred_boxes"):
            det["bbox_xyxy"] = _to_builtin(pred_instances.pred_boxes[i])
        if pred_instances.has("scores"):
            det["score"] = float(_to_builtin(pred_instances.scores[i]))
        if pred_instances.has("pred_classes"):
            det["class_id"] = int(_to_builtin(pred_instances.pred_classes[i]))
        dets.append(det)

    payload = {
        "video_id": int(video_id) if video_id is not None else None,
        "timestamp": float(timestamp) if timestamp is not None else None,
        "image_size_hw": [int(image_h), int(image_w)],
        "detections": dets,
    }

    # Optional: 3D boxes (if present) as well.
    if pred_instances is not None and pred_instances.has("pred_boxes_3d"):
        boxes3d = pred_instances.pred_boxes_3d
        payload["boxes_3d"] = {
            # In this codebase, gravity_center is (x,y,z); dims is (l,h,w) for GeneralInstance3DBoxes.tensor[:,3:6].
            "gravity_center_xyz": _to_builtin(boxes3d.gravity_center),
            "dims_lhw": _to_builtin(boxes3d.dims),
            "R_3x3": _to_builtin(boxes3d.R),
        }

    stem = _safe_frame_stem(video_id if video_id is not None else 0, timestamp if timestamp is not None else 0.0)
    out_path = os.path.join(out_dir, f"{stem}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_data_and_execute_model(model, dataset, augmentor, preprocessor, score_thresh=0.0, viz_on_gt_points=False, save_preds_dir=None):
    is_depth_model = "wide/depth" in augmentor.measurement_keys
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            contents=[
                rrb.Spatial3DView(
                    name="World",
                    contents=[
                        "+ $origin/**",
                        "+ /device/wide/pred_instances/**"
                    ],
                    origin="/world"),
                rrb.Horizontal(
                    contents=([
                        rrb.Spatial2DView(
                            name="Image",
                            origin="/device/wide/image",
                            contents=[
                                "+ $origin/**",
                                "+ /device/wide/pred_instances/**"
                            ])
                    ] + ([
                        # Only show this for RGB-D.
                        rrb.Spatial2DView(
                            name="Depth",
                            origin="/device/wide/depth")
                    ] if is_depth_model else [])),
                    name="Wide")
            ]))

    recording = None
    video_id = None

    device = model.pixel_mean
    for sample in dataset:
        sample_video_id = sample["meta"]["video_id"]
        if (recording is None) or (video_id != sample_video_id):
            new_recording = rerun.new_recording(
                application_id=str(sample_video_id), recording_id=uuid.uuid4(), make_default=True)
            new_recording.send_blueprint(blueprint, make_active=True)
            rerun.spawn()

            recording = new_recording
            video_id = sample_video_id

            # Keep things in image space, so adjust accordingly.
            rerun.log("/world", rerun.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True) 

        rerun.set_time_seconds("pts", sample["meta"]["timestamp"], recording=recording)

        # -> channels last.
        image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)        
        color_camera = rerun.Pinhole(
            image_from_camera=sample["sensor_info"].wide.image.K[-1].numpy(), resolution=sample["sensor_info"].wide.image.size)

        if is_depth_model:
            # Show the depth being sent to the model.            
            depth_camera = rerun.Pinhole(
                image_from_camera=sample["sensor_info"].wide.depth.K[-1].numpy(), resolution=sample["sensor_info"].wide.depth.size)

        xyzrgb = None
        if viz_on_gt_points and sample["sensor_info"].has("gt"):
            # Backproject GT depth to world so we can compare our predictions.
            depth_gt = sample["gt"]["depth"][-1]
            matched_image = torch.tensor(np.array(Image.fromarray(image).resize((depth_gt.shape[1], depth_gt.shape[0]))))

            # Feel free to change max_depth, but know CA is only trained up to 5m.
            xyz, valid = unproject(depth_gt, sample["sensor_info"].gt.depth.K[-1], torch.eye(4), max_depth=10.0)
            xyzrgb = torch.cat((xyz, matched_image / 255.0), dim=-1)[valid]            
                    
        packaged = augmentor.package(sample)
        packaged = move_input_to_current_device(packaged, device)
        packaged = preprocessor.preprocess([packaged])

        with torch.no_grad():
            pred_instances = model(packaged)[0]

        pred_instances = pred_instances[pred_instances.scores >= score_thresh]

        if save_preds_dir:
            save_pred_bboxes_json(pred_instances, sample.get("meta", {}), save_preds_dir)
        
        # Hold off on logging anything until now, since the delay might confuse the user in the visualizer.
        rerun.log("/device/wide/image", rerun.Image(image).compress())
        rerun.log("/device/wide/image", color_camera)

        if is_depth_model:
            rerun.log("/device/wide/depth", rerun.DepthImage(sample["wide"]["depth"][-1].numpy()))
            rerun.log("/device/wide/depth", depth_camera)
        
        if xyzrgb is not None:
            rerun.log("/world/xyz", rerun.Points3D(positions=xyzrgb[..., :3], colors=xyzrgb[..., 3:], radii=None))        

        log_instances(pred_instances, prefix="/device/wide", boxes_3d_name="pred_boxes_3d", ids_name=None, log_instances_name="pred_instances")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", help="Path to the directory containing the .tar files, the full path to a single tar file (recommended), or a path to a txt file containing HTTP links. Using the value \"stream\" will attempt to stream from your device using the NeRFCapture app")
    parser.add_argument("--model-path", help="Path to the model to load")
    parser.add_argument("--no-depth", default=False, action="store_true", help="Skip loading depth.")
    parser.add_argument("--score-thresh", default=0.25, help="Threshold for detections")
    parser.add_argument("--every-nth-frame", default=None, type=int, help="Load every `n` frames")
    parser.add_argument("--viz-only", default=False, action="store_true", help="Skip loading a model and only visualize data.")
    parser.add_argument("--viz-on-gt-points", default=False, action="store_true", help="Backproject the GT depth to form a point cloud in order to visualize the predictions")
    parser.add_argument("--device", default="cpu", help="Which device to push the model to (cpu, mps, cuda)")
    parser.add_argument("--video-ids", nargs="+", help="Subset of videos to execute on. By default, all. Ignored if a tar file is explicitly given or in stream mode.")
    parser.add_argument("--save-preds-dir", default=None, help="If set, saves a JSON per frame with predicted 2D boxes/scores/classes (and 3D boxes if available).")

    args = parser.parse_args()
    print("Command Line Args:", args)

    dataset_path = args.dataset_path
    use_cache = False
    
    if dataset_path == "stream":
        dataset = CaptureDataset()
    else:
        dataset_files = []

        # Allow the user to specify a single tar or a txt file containing an http link per line.
        if os.path.isfile(dataset_path):
            if dataset_path.endswith(".txt"):
                with open(dataset_path, "r") as dataset_file:
                    dataset_files = [l.strip() for l in dataset_file.readlines()]

                # Cache these files locally to prevent repeated downlods.
                use_cache = True
            else:
                args.video_ids = None
                dataset_files = [dataset_path]
        else:
            # Try to glob all files matching ca1m-*.tar
            dataset_files = glob.glob(os.path.join(dataset_path, "ca1m-*.tar"))
            if len(dataset_files) == 0:
                raise ValueError(f"Failed to find any .tar files matching ca1m- prefix at {dataset_path}")

        if args.video_ids is not None:
            dataset_files = [df for df in dataset_files if Path(df).with_suffix("").name.split("-")[-1] in args.video_ids]

        if len(dataset_files) == 0:
            raise ValueError("No data was found")
            
        dataset = CubifyAnythingDataset(
            [Path(df).as_uri() if not df.startswith("https://") else df for df in dataset_files],
            yield_world_instances=args.viz_only,
            load_arkit_depth=not args.no_depth,
            use_cache=use_cache)

    if args.viz_only:
        if args.every_nth_frame is not None:
            dataset = itertools.islice(dataset, 0, None, args.every_nth_frame)
        
        load_data_and_visualize(dataset)
        sys.exit(0)

    assert args.model_path is not None
    checkpoint = torch.load(args.model_path, map_location=args.device or "cpu")["model"]

    # Figure out which model this is based on the weights.

    # Basic detection of the actual ViT backbone being used (for our setup, dimension is 1:1 with which ViT).
    backbone_embedding_dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
        
    # We need to detect RGB or RGB only models so we can disable sending depth.                
    is_depth_model = any(k.startswith("backbone.0.patch_embed_depth.") for k in checkpoint.keys())

    model = make_cubify_transformer(dimension=backbone_embedding_dimension, depth_model=is_depth_model).eval()
    model.load_state_dict(checkpoint)

    # No need for ARKit depth if running an RGB only model.
    dataset.load_arkit_depth = is_depth_model
    if args.every_nth_frame is not None:
        dataset = itertools.islice(dataset, 0, None, args.every_nth_frame)

    augmentor = Augmentor(("wide/image", "wide/depth") if is_depth_model else ("wide/image",))
    preprocessor = Preprocessor()
        
    if args.device is not None:
        model = model.to(args.device)

    load_data_and_execute_model(
        model,
        dataset,
        augmentor,
        preprocessor,
        score_thresh=args.score_thresh,
        viz_on_gt_points=args.viz_on_gt_points,
        save_preds_dir=args.save_preds_dir,
    )
