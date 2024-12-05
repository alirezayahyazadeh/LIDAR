import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim


# Define the PointPillars Model
class PointPillars(nn.Module):
    def __init__(self):
        super(PointPillars, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Linear(256 * 64 * 64, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Configure Intel RealSense Pipeline
def configure_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


# Preprocess depth and color frames into a point cloud
def preprocess_frames(depth_frame, color_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    # Create Open3D point cloud
    pc = o3d.geometry.PointCloud()
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    points = []
    for y in range(depth_image.shape[0]):
        for x in range(depth_image.shape[1]):
            depth = depth_image[y, x]
            if depth != 0:
                points.append(rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth))
    pc.points = o3d.utility.Vector3dVector(np.array(points))
    return pc


# Run the pipeline and model inference
def run_pointpillars(pipeline, model):
    try:
        while True:
            # Capture frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Preprocess frames to generate a point cloud
            point_cloud = preprocess_frames(depth_frame, color_frame)
            
            # Apply a distance threshold to filter points
            distance_threshold = 1.5  # 1.5 meters
            point_cloud = point_cloud.select_by_index(
                np.where(np.asarray(point_cloud.points)[:, 2] < distance_threshold)[0]
            )

            # Generate BEV data
            bev_data, _ = np.histogramdd(
                np.asarray(point_cloud.points), bins=(64, 64, 64)
            )
            
            # Sum along one axis (e.g., Z-axis) to flatten 3D data into 2D
            bev_data = bev_data.sum(axis=2)  # Shape becomes [64, 64]

            # Prepare tensor input for the model
            bev_data = torch.tensor(bev_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 64, 64]

            # Perform inference
            detections = model(bev_data)
            print("Detections:", detections)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    # Initialize model
    model = PointPillars()

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initialize RealSense pipeline
    pipeline = configure_realsense()

    # Run the system
    run_pointpillars(pipeline, model)
