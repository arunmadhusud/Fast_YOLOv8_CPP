#include "inference.h"

#include <iostream>
#include <opencv2/highgui.hpp>


void Detector(OpenVinoInference::Inference& inference, const std::string& img_path, const std::string& video_path, bool video=true) {
    if (video) {
        // Open the video file
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "ERROR: Could not open video file" << std::endl;
            return;
        }

        cv::Mat frame, resizedFrame;
        while (cap.read(frame)) {
            if (frame.empty()) break;

            // Run inference on the resized frame
            inference.RunInference(frame);

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
        return;
    } else {
        // Read the input image
        cv::Mat image = cv::imread(img_path);
        
        // Check if the image was successfully loaded
        if (image.empty()) {
            std::cerr << "ERROR: image is empty" << std::endl;
            return;
        }
        
        // Run inference on the input image
        inference.RunInference(image);
        
        // Display the image with the detections
        cv::imshow("image", image);
        cv::waitKey(0);

        return;
    }
}

int main(int argc, char **argv) {
    const std::string model_path = "/home/arun/Adv_CV/onnx_rt/yolo_open_vino/yolov8n_int8.xml";
    const std::string video_path = "/home/arun/Adv_CV/onnx_rt/yolo_onnxruntime/video/test.mp4";
    const std::string img_path = "/home/arun/Adv_CV/onnx_rt/yolo_onnxruntime/images/bus.jpg";


    bool video = true;

    // Define the confidence and NMS thresholds
    const float confidence_threshold = 0.3;
    const float NMS_threshold = 0.5;

    // Initialize the YOLO inference with the specified model and parameters
    OpenVinoInference::Inference inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);

    // Call the Detector function
    Detector(inference, img_path, video_path, video);

    inference.avgFPS = static_cast<int>(inference.totalFPS / inference.frameCount);
    std::cout << "Average FPS: " << inference.avgFPS << std::endl;

    return 0;
}



