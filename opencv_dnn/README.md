# YOLOv8n cvDNN Inference  

## Introduction  
This repository contains the code for inferencing YOLOv8n in C++ using `cv::dnn` with the Intel OpenVINO backend. The models loaded are in OpenVINO™ Model Optimizer format (`.bin` and `.xml`).  You can use both FP32 models or static INT8 quantized models. The provided Jupyter notebook allows exporting and quantizing the models. Tested in following environments:    
- Ubuntu 20.04  
- OpenCV 4.12  
- OpenVINO 2024.6.0  

## Installation  

1. Install OpenVINO using the official instructions:  
   [OpenVINO Installation Guide](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html)  

2. Build OpenCV from source with `DWITH_OPENVINO=ON`:  
   - It is recommended to use the latest OpenCV version for best performance.  
   - Before building OpenCV, source the OpenVINO environment variables:  
     ```bash
     source /opt/intel/openvino_2024/setupvars.sh
     ```  

## Usage  

### Build Instructions  

1. Clone the repository:  
   ```bash
    git clone https://github.com/arunmadhusud/Fast_YOLOv8_CPP
    cd opencv_dnn
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