# **PointPillars for 3D Point Cloud Processing**  
This repository contains an implementation of a PointPillars model integrated with Intel RealSense cameras for processing point clouds and performing 3D object detection.

---

## **Project Overview**  
The purpose of this project is to:  
- Capture real-time 3D data from Intel RealSense cameras.  
- Preprocess the data into a Bird’s Eye View (BEV) representation.  
- Run it through a neural network model for object detection or classification.

### **Key Components**  
- **PointPillars Neural Network Model**: A custom PyTorch-based model for processing BEV data.  
- **Intel RealSense Integration**: Captures depth and RGB data, converting it into a 3D point cloud.  
- **Real-Time Inference**: Performs object detection in real-time on processed data.  

---

## **Features**  
- Real-time depth and color data capture using Intel RealSense.  
- 3D point cloud generation and filtering using Open3D.  
- Bird’s Eye View (BEV) projection of point cloud data.  
- A simple neural network model for classification and detection tasks.  

---

## **File Description**  

### **`pointpillar_5.py`**  
The main Python script containing:  
- PointPillars model definition.  
- Integration with Intel RealSense cameras.  
- Real-time preprocessing of 3D point clouds into BEV.  
- Real-time inference with a PyTorch model.  

---

## **Prerequisites**  
Ensure you have the following dependencies installed:  

- **Python**: Version 3.8+  
- **Intel RealSense SDK**  
- **PyTorch**  
- **Open3D**  
- **NumPy**  
- Other libraries: `torch.optim`, `torch.nn`  

---

## **Installation**  

### **Clone the Repository**  
```bash
git clone https://github.com/alirezayahyazadeh/RealSense PointPillars: Real-Time 3D Object Detection Framework
