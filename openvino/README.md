# YOLOv8n OpenVINO Inference

## Introduction
This repository contains the code for inferencing YOLOv8n in C++ using OpenVINO library. You can use both FP32 models or static INT8 quantized models. You can export and quantize the models using the jupyter notebook provided. Tested in following environments:
- Ubuntu 20.04
- OpenCV 4.12
- OpenVINO 2024.6.0

## Installation
Install OpenVINO using the instructions from the official documentation:
[OpenVINO Installation Guide](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html)

## Usage
### Build Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/arunmadhusud/Fast_YOLOv8_CPP
    cd openvino
    ```

2. Source the OpenVINO environment variables:
    ```bash
    source /opt/intel/openvino_2024/setupvars.sh
    ```

3. Create a build directory, build the project and run the inference:
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

