#include <iostream>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "inference.h"

using namespace std;
using namespace cv;


void Detector(CvdnnInference::Inference& inf, const std::string& img_path, const std::string& video_path, bool video=true) {
    if (!video) {
        cv::Mat frame = cv::imread(img_path);
        if (frame.empty()) {
            std::cerr << "Error loading image" << std::endl;
            return;
        }

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::vector<CvdnnInference::Detection> output = inf.runInference(frame);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        double inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        int fps_ = static_cast<int>(1000.0 / inference_time);

        std::cout << "Inference time = " << inference_time  << "[ms]" << std::endl;
        std::cout << "FPS = " << fps_ << "[fps]" << std::endl;

        std::string fpsText = "FPS: " + std::to_string(fps_);
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, 8);
        cv::imshow("Inference", frame);
        cv::waitKey(0);
        return;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return;
    }

    cv::Mat frame,resizedFrame;
    double totalTime = 0.0;
    int frameCount = 0;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::vector<CvdnnInference::Detection> output = inf.runInference(frame);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        double inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        int fps_ = static_cast<int>(1000.0 / inference_time);

        // Update total time and frame count
        totalTime += inference_time;
        frameCount++;

        // Compute average FPS
        double avgFPS = (frameCount / totalTime) * 1000.0;

        std::string fpsText = "FPS: " + std::to_string(fps_);
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, 8);

        // Resize only for display purposes
        cv::Mat displayFrame;
        cv::resize(frame, displayFrame, cv::Size(1280, 720));
        
        // Display the smaller frame with detections
        cv::imshow("video", displayFrame);

        if (cv::waitKey(1) == 27) break; // ESC key
    }
    std::cout << "Final Average FPS: " << (frameCount / totalTime) * 1000.0 << std::endl;

    cap.release();
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    const cv::String  modelConfig = "/home/arun/Adv_CV/onnx_rt/yolo_cv_dnn/yolov8n_int8.xml";
    const cv::String  modelWeights = "/home/arun/Adv_CV/onnx_rt/yolo_cv_dnn/yolov8n_int8.bin";

    CvdnnInference::Inference inf(modelConfig,modelWeights, cv::Size(640, 640));

    std::string img_path = "/home/arun/Adv_CV/onnx_rt/yolo_cv_dnn/bus.jpg";
    std::string video_path = "/home/arun/Adv_CV/onnx_rt/yolo_onnxruntime/video/test.mp4";
    bool processVideo = true; // Change to true for video processing

    Detector(inf, img_path, video_path, processVideo);
    return 0;
}

