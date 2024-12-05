import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
import cv2

# Camera Intrinsic Parameters
class CameraIntrinsics:
    def __init__(self, fx, fy, ppx, ppy):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy

# Initialize Intel RealSense Camera
def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align

# Capture Frames
def process_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return depth_image, color_image

# Remove Background
def remove_background(color_image, depth_image, threshold=1000):
    mask = (depth_image > 0) & (depth_image < threshold)
    color_image[~mask] = 0
    return color_image

# Create Point Cloud
def create_point_cloud(color_image, depth_image, intrinsics):
    height, width = depth_image.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    x = np.tile(np.arange(width), height).reshape(-1)
    y = np.repeat(np.arange(height), width).reshape(-1)
    z = depth_image.flatten() / 1000.0  # Convert to meters

    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]

    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    points = np.stack((x, y, z), axis=-1)
    colors = color_image.reshape(-1, 3)[valid]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)
    return point_cloud

# Load PointPillars Model
def load_pointpillars_model(config_path, checkpoint_path):
    # Load configuration
    cfg_from_yaml_file(config_path, cfg)

    # Build the PointPillars model
    model = build_network(model_cfg=cfg.MODEL, num_class=cfg.DATA_CONFIG.NUM_CLASS, dataset=None)

    # Load pre-trained weights
    model.load_params_from_file(checkpoint_path, logger=None)
    model.cuda()
    model.eval()
    return model

# Detect Objects
def detect_objects(model, point_cloud):
    points = np.asarray(point_cloud.points)
    if points.shape[1] < 4:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))  # Add intensity feature

    data_dict = {'points': points, 'frame_id': 0, 'metadata': None}
    load_data_to_gpu(data_dict)
    pred_dicts, _ = model.forward(data_dict)
    return pred_dicts

# Main Function
def main():
    # Initialize Intel RealSense Camera
    pipeline, align = init_camera()

    # Camera Intrinsics (Adjust these values based on your RealSense device)
    intrinsics = CameraIntrinsics(fx=615.0, fy=615.0, ppx=320.0, ppy=240.0)

    # Paths for Configuration and Model Weights
    config_path = './OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml'
    checkpoint_path = '/mnt/data/pointpillar_7728.pth'  # Uploaded model file path

    # Load the PointPillars Model
    pointpillars_model = load_pointpillars_model(config_path, checkpoint_path)

    try:
        while True:
            # Capture frames from the RealSense camera
            depth_image, color_image = process_frames(pipeline, align)
            if depth_image is None or color_image is None:
                continue

            # Remove background based on depth threshold
            filtered_color_image = remove_background(color_image, depth_image)

            # Generate a Point Cloud
            point_cloud = create_point_cloud(filtered_color_image, depth_image, intrinsics)

            # Detect objects using PointPillars
            results = detect_objects(pointpillars_model, point_cloud)

            # Print Detected Objects
            print("Detections:", results)

            # Optional: Visualize the Point Cloud
            o3d.visualization.draw_geometries([point_cloud])

            # Show filtered color image
            cv2.imshow('Filtered Image', filtered_color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
