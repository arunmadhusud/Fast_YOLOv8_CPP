# YOLOv8n ONNX Runtime Inference  

## Introduction  
This repository contains the code for inferencing YOLOv8n in C++ using ONNX Runtime** with OpenVINO as the execution provider.  You can use both FP32 models or static INT8 quantized models. The provided Jupyter notebook allows exporting and quantizing the models.  Tested Environments:
- Ubuntu 20.04  
- OpenCV 4.12  
- OpenVINO 2024.6.0  
- ONNX Runtime v1.20.1  

## Installation  

1. Install OpenVINO using the official instructions:  
   [OpenVINO Installation Guide](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html)  

2. Build ONNX Runtime for inference using the official instructions:  
   [ONNX Runtime Installation Guide](https://onnxruntime.ai/docs/build/)  
   - Before building, source the OpenVINO environment variables:  
     ```bash
     source /opt/intel/openvino_2024/setupvars.sh
     ```  
   - Build ONNX Runtime with OpenVINO support:  
     ```bash
     ./build.sh --config RelWithDebInfo --use_openvino CPU --build_shared_lib --build_wheel
     ```

## Usage  

### Build Instructions  

1. Clone the repository:  
    ```bash
    git clone https://github.com/arunmadhusud/Fast_YOLOv8_CPP
    cd onnxruntime
    ```

2. Source the OpenVINO environment variables:  
    ```bash
    source /opt/intel/openvino_2024/setupvars.sh
    ```

3. Create a build directory, compile the project, and run inference:  
    ```bash
    mkdir build && cd build
    cmake ..
    make 
    ./inference
    ```

## Result  
Sample inference result:  

![Inference Result](sample.gif)  

## Acknowledgment  
This project uses models from Ultralytics YOLOv8 implementation.  
