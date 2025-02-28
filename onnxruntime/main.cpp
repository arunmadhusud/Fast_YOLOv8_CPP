#include <iostream>
#include <iomanip>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>

void Detector(OnnxRuntimeInference::YOLO_V8& yoloDetector, const std::string& img_path, const std::string& video_path, bool video=true) {
    if (!video) {
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) {
            std::cerr << "Error loading image" << std::endl;
            return;
        }
        
        std::vector<OnnxRuntimeInference::DL_RESULT> res;
        yoloDetector.RunSession(img, res);        
        cv::imshow("Inference", img);
        cv::waitKey(0);
        return;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return;
    }

    cv::Mat frame, resizedFrame;
    while (cap.read(frame)) {
        std::vector<OnnxRuntimeInference::DL_RESULT> res;
        yoloDetector.RunSession(frame, res);
        
        // Resize only for display purposes
        cv::Mat displayFrame;
        cv::resize(frame, displayFrame, cv::Size(1280, 720));
        
        // Display the smaller frame with detections
        cv::imshow("video", displayFrame);
        
        if (cv::waitKey(1) == 27) {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    OnnxRuntimeInference::YOLO_V8 yoloDetector;
    OnnxRuntimeInference::DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.3;
    params.iouThreshold = 0.5;
    params.modelPath = "/home/arun/Adv_CV/onnx_rt/models/yolov8n_st_quant.onnx";
    params.imgSize = {640, 640};
    yoloDetector.CreateSession(params);

    std::string img_path = "/home/arun/Adv_CV/onnx_rt/yolo_onnxruntime/images/bus.jpg";
    std::string video_path = "/home/arun/Adv_CV/onnx_rt/yolo_onnxruntime/video/test.mp4";
    bool processVideo = true;
    
    Detector(yoloDetector, img_path, video_path, processVideo);
    yoloDetector.avgFPS = static_cast<int>(yoloDetector.totalFPS / yoloDetector.frameCount);
    std::cout << "Average FPS: " << yoloDetector.avgFPS << std::endl;


    return 0;
}



