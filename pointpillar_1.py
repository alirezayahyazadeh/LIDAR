import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader
from  pointpillar import PointPillars  # Assume PointPillars is a local module or package
import cv2

# Initialize Intel RealSense camera
def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align

# Function to process depth and color frames
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

# Remove background based on depth threshold
def remove_background(color_image, depth_image, threshold=1000):
    mask = (depth_image > 0) & (depth_image < threshold)
    color_image[~mask] = 0
    return color_image

# Convert depth and color to Open3D Point Cloud
def create_point_cloud(color_image, depth_image, camera_intrinsics):
    height, width = depth_image.shape
    fx, fy = camera_intrinsics.fx, camera_intrinsics.fy
    cx, cy = camera_intrinsics.ppx, camera_intrinsics.ppy

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

# Load PointPillars model
def load_pointpillars_model(model_path):
    model = PointPillars()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Perform object detection
def detect_objects(model, point_cloud):
    points = np.asarray(point_cloud.points, dtype=np.float32)
    data_loader = DataLoader([points], batch_size=1, shuffle=False)
    results = []

    with torch.no_grad():
        for data in data_loader:
            result = model(data)
            results.append(result)

    return results

# Main function
def main():
    pipeline, align = init_camera()

    # Replace with actual intrinsics from your RealSense camera
    class CameraIntrinsics:
        fx = 640
        fy = 480
        ppx = 320
        ppy = 240
    camera_intrinsics = CameraIntrinsics()

    # Load PointPillars model
    model_path = 'pointpillars_model.pth'  # Replace with your model path
    pointpillars_model = load_pointpillars_model(model_path)

    try:
        while True:
            depth_image, color_image = process_frames(pipeline, align)
            if depth_image is None or color_image is None:
                continue

            # Remove background
            filtered_color_image = remove_background(color_image, depth_image)

            # Generate point cloud
            point_cloud = create_point_cloud(filtered_color_image, depth_image, camera_intrinsics)

            # Detect objects using PointPillars
            results = detect_objects(pointpillars_model, point_cloud)

            # Visualize point cloud (Optional)
            o3d.visualization.draw_geometries([point_cloud])

            # Print detected objects
            for result in results:
                print("Detected object:", result)

            cv2.imshow('Filtered Image', filtered_color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
