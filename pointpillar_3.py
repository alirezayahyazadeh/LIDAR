import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim

class PointPillars(nn.Module):
    def init(self):  # اصلاح شده: init
        super(PointPillars, self).init()
        self.encoder = nn.Sequential(
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

def configure_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def preprocess_frames(depth_frame, color_frame):
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
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

def run_pointpillars(pipeline, model):
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            point_cloud = preprocess_frames(depth_frame, color_frame)
            distance_threshold = 1.5
            point_cloud = point_cloud.select_by_index(
                np.where(np.asarray(point_cloud.points)[:, 2] < distance_threshold)[0]
            )

            bev_data = np.histogramdd(
                np.asarray(point_cloud.points), bins=(64, 64, 64)
            )[0]
            bev_data = torch.tensor(bev_data, dtype=torch.float32).unsqueeze(0)

            detections = model(bev_data)
            print("Detections:", detections)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    model = PointPillars()
    pipeline = configure_realsense()

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # خطا باید رفع شود
    criterion = nn.CrossEntropyLoss()

    run_pointpillars(pipeline, model)