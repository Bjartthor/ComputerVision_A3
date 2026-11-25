// Run from the code directory after exporting the accompanying ONNX model.
#include "dnn_engine.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

static const std::array<const char*, 80> classes = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

int main(int argc, char** argv) try {
    opencv5_example::checkEnvironment();
    cv::CommandLineParser parser(argc, argv,
        "{help h||Show usage}"
        "{model|yolo26/yolo26n.onnx|End-to-end ONNX export}"
        "{image|yolo26/bus.jpg|Input photograph}"
        "{output|yolo26/yolo26_cpp_result.jpg|Annotated image}"
        "{confidence|0.35|Confidence threshold}"
        "{details||Optional JSON detections for validation}");
    if (parser.has("help")) { parser.printMessage(); return 0; }
    const auto modelPath = parser.get<std::string>("model");
    const auto imagePath = parser.get<std::string>("image");
    const auto outputPath = parser.get<std::string>("output");
    const auto detailsPath = parser.get<std::string>("details");
    const float threshold = parser.get<float>("confidence");
    if (!parser.check()) { parser.printErrors(); return 1; }
    if (!(threshold >= 0 && threshold <= 1))
        throw std::runtime_error("Confidence must be between 0 and 1.");
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) throw std::runtime_error("Cannot read image: " + imagePath);

    // Letterbox to 640 x 640, matching the Python example.
    constexpr int size = 640;
    const double gain = std::min(double(size) / image.cols, double(size) / image.rows);
    // nearbyint uses ties-to-even rounding, as Python round() does.
    const int newWidth = int(std::nearbyint(image.cols * gain));
    const int newHeight = int(std::nearbyint(image.rows * gain));
    const int left = (size - newWidth) / 2, top = (size - newHeight) / 2;
    cv::Mat resized, padded;
    cv::resize(image, resized, cv::Size(newWidth, newHeight));
    cv::copyMakeBorder(resized, padded, top, size - newHeight - top,
        left, size - newWidth - left, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0,
        cv::Size(), cv::Scalar(), true, false, CV_32F);
    auto net = opencv5_example::readNet(modelPath);
    net.setInput(blob);
    cv::Mat output = net.forward();
    if (output.dims != 3 || output.size[0] != 1 || output.size[1] != 300 ||
        output.size[2] != 6 || output.type() != CV_32F)
        throw std::runtime_error("Expected float32 output [1, 300, 6]; use the supplied end-to-end export.");

    std::ofstream details;
    if (!detailsPath.empty()) {
        const auto parent = std::filesystem::path(detailsPath).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        details.open(detailsPath);
        if (!details) throw std::runtime_error("Cannot write details JSON.");
        details << std::setprecision(9) << "{\"opencv\":\"" << CV_VERSION
            << "\",\"engine\":\"" << opencv5_example::engineName
            << "\",\"output_shape\":[1,300,6],\"detections\":[";
    }

    int count = 0;
    for (int i = 0; i < output.size[1]; ++i) {
        const float* row = output.ptr<float>(0, i);
        if (row[4] < threshold) continue;
        const int classId = int(row[5]);
        if (classId < 0 || classId >= int(classes.size()))
            throw std::runtime_error("Unexpected class ID in model output.");
        const float x1 = std::clamp(float((row[0] - left) / gain), 0.0f, float(image.cols));
        const float y1 = std::clamp(float((row[1] - top) / gain), 0.0f, float(image.rows));
        const float x2 = std::clamp(float((row[2] - left) / gain), 0.0f, float(image.cols));
        const float y2 = std::clamp(float((row[3] - top) / gain), 0.0f, float(image.rows));
        if (x2 <= x1 || y2 <= y1) continue;
        const cv::Point p1(int(std::nearbyint(x1)), int(std::nearbyint(y1)));
        const cv::Point p2(int(std::nearbyint(x2)), int(std::nearbyint(y2)));
        const auto label = cv::format("%s %.2f", classes[classId], row[4]);
        cv::rectangle(image, p1, p2, cv::Scalar(40, 220, 40), 2);
        cv::putText(image, label, cv::Point(p1.x, std::max(p1.y - 8, 22)),
            cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(40, 220, 40), 2, cv::LINE_AA);
        std::cout << label << '\n';
        if (details.is_open()) {
            if (count) details << ',';
            details << "{\"class_id\":" << classId << ",\"score\":" << row[4]
                << ",\"box\":[" << x1 << ',' << y1 << ',' << x2 << ',' << y2 << "]}";
        }
        ++count;
    }
    if (details.is_open()) { details << "]}\n"; details.close(); }
    const auto parent = std::filesystem::path(outputPath).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    if (!cv::imwrite(outputPath, image)) throw std::runtime_error("Could not write output image.");
    std::cout << "OpenCV " << CV_VERSION << " / " << opencv5_example::engineName
        << ": saved " << count << " detections to " << outputPath << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
