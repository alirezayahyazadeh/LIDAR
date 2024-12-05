import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu

# Camera Intrinsic Parameters Class
class CameraIntrinsics:
    def __init__(self, fx, fy, ppx, ppy):
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy

# Initialize RealSense Camera
def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align

# Get Frames from RealSense Camera
def get_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return depth_image, color_image

# Remove Background Based on Depth Threshold
def remove_background(color_image, depth_image, threshold=1000):
    mask = (depth_image > 0) & (depth_image < threshold)
    color_image[~mask] = 0
    return color_image

# Create Point Cloud from Depth and Color Frames
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
    cfg_from_yaml_file(config_path, cfg)
    model = build_network(model_cfg=cfg.MODEL, num_class=cfg.DATA_CONFIG.NUM_CLASS, dataset=None)
    model.load_params_from_file(checkpoint_path, logger=None)
    model.cuda()
    model.eval()
    return model

# Run Object Detection with PointPillars
def detect_objects(model, point_cloud):
    points = np.asarray(point_cloud.points)
    # Append zeros as dummy intensities if required by PointPillars
    if points.shape[1] < 4:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))
    data_dict = {'points': points, 'frame_id': 0, 'metadata': None}
    load_data_to_gpu(data_dict)
    pred_dicts, _ = model.forward(data_dict)
    return pred_dicts

# Main Function
def main():
    # Initialize RealSense camera
    pipeline, align = init_camera()

    # Replace these values with your camera intrinsics
    intrinsics = CameraIntrinsics(fx=615.0, fy=615.0, ppx=320.0, ppy=240.0)

    # Paths to PointPillars config and checkpoint files
    config_path = './OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml'
    checkpoint_path = './OpenPCDet/weights/pointpillar.pth'

    # Load the PointPillars model
    model = load_pointpillars_model(config_path, checkpoint_path)

    try:
        while True:
            # Capture frames from the camera
            depth_image, color_image = get_frames(pipeline, align)
            if depth_image is None or color_image is None:
                continue

            # Remove background from the color image
            filtered_color_image = remove_background(color_image, depth_image)

            # Generate a point cloud
            point_cloud = create_point_cloud(filtered_color_image, depth_image, intrinsics)

            # Perform object detection with PointPillars
            detections = detect_objects(model, point_cloud)

            # Print detection results
            print("Detections:", detections)

            # Visualize the point cloud (Optional)
            o3d.visualization.draw_geometries([point_cloud])

    except KeyboardInterrupt:
        print("Exiting program.")
    finally:
        pipeline.stop()

# Run the script
if __name__ == "__main__":
    main()
