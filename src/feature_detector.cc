#include <opencv2/opencv.hpp>

#include "plnet.h"
#include "feature_detector.h"
#include "utils.h"

FeatureDetector::FeatureDetector(const PLNetConfig& plnet_config) : _plnet_config(plnet_config){
  if(_plnet_config.use_superpoint){
    SuperPointConfig superpoint_config;
    superpoint_config.max_keypoints = plnet_config.max_keypoints;
    superpoint_config.keypoint_threshold = plnet_config.keypoint_threshold;
    superpoint_config.remove_borders = plnet_config.remove_borders;
    superpoint_config.dla_core = -1;

    superpoint_config.input_tensor_names.push_back("input");
    superpoint_config.output_tensor_names.push_back("scores");
    superpoint_config.output_tensor_names.push_back("descriptors");

    superpoint_config.onnx_file = plnet_config.superpoint_onnx;
    superpoint_config.engine_file = plnet_config.superpoint_engine;

    _superpoint = std::shared_ptr<SuperPoint>(new SuperPoint(superpoint_config));
    if (!_superpoint->build()){
      std::cout << "Error in SuperPoint building" << std::endl;
      exit(0);
    }
  }

  YoloConfig yolo_config;

  yolo_config.dla_core = -1;

  yolo_config.input_tensor_names.push_back("images");
  yolo_config.output_tensor_names.push_back("output0");


  yolo_config.onnx_file = "/home/lan/Airslam_ws/src/AirSLAM/output/yolov8n.onnx";
  yolo_config.engine_file = "/home/lan/Airslam_ws/src/AirSLAM/output/yolov8n.engine";

  _yolodetecter = std::shared_ptr<YoloDetecter>(new YoloDetecter(yolo_config));
  if (!_yolodetecter->build()){
    std::cout << "Error in _yolodetecter building" << std::endl;
    exit(0);
  }

  _plnet = std::shared_ptr<PLNet>(new PLNet(_plnet_config));
  if (!_plnet->build()){
    std::cout << "Error in FeatureDetector building" << std::endl;
    // exit(0);
  }
}

bool FeatureDetector::Detect(cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features){
  bool good_infer = false;
  if(_plnet_config.use_superpoint){
    good_infer = _superpoint->infer(image, features);
  }else{
    std::vector<Eigen::Vector4d> lines;
    good_infer = Detect(image, features, lines);
  }


  if(!good_infer){
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer; 
}

bool FeatureDetector::Detect(cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
    std::vector<Eigen::Vector4d>& lines){
  Eigen::Matrix<float, 259, Eigen::Dynamic> junctions;
  bool good_infer = _plnet->infer(image, features, lines, junctions);
  if(!good_infer){
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer; 
}

bool FeatureDetector::Detect(cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
    std::vector<Eigen::Vector4d>& lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions){
  bool good_infer = _plnet->infer(image, features, lines, junctions, true);
  if(!good_infer){
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer; 
}

bool FeatureDetector::Detect(cv::Mat& image_left, cv::Mat& image_right, 
    Eigen::Matrix<float, 259, Eigen::Dynamic> & left_features, 
    Eigen::Matrix<float, 259, Eigen::Dynamic> & right_features){
  bool good_infer_left = Detect(image_left, left_features);
  bool good_infer_right = Detect(image_right, right_features);
  bool good_infer = good_infer_left & good_infer_right;
  if(!good_infer){
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer; 
}

bool FeatureDetector::Detect(cv::Mat& image_left, cv::Mat& image_right, 
    Eigen::Matrix<float, 259, Eigen::Dynamic> & left_features, 
    Eigen::Matrix<float, 259, Eigen::Dynamic> & right_features, 
    std::vector<Eigen::Vector4d>& left_lines, 
    std::vector<Eigen::Vector4d>& right_lines){
  bool good_infer_left = Detect(image_left, left_features, left_lines);
  bool good_infer_right = Detect(image_right, right_features, right_lines);
  bool good_infer = good_infer_left & good_infer_right;
  if(!good_infer){
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer; 
}

bool FeatureDetector::Detect(cv::Mat& image_left, cv::Mat& image_right, Eigen::Matrix<float, 259, Eigen::Dynamic> & left_features, 
    Eigen::Matrix<float, 259, Eigen::Dynamic> & right_features, std::vector<Eigen::Vector4d>& left_lines, 
    std::vector<Eigen::Vector4d>& right_lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions){
  bool good_infer_left = Detect(image_left, left_features, left_lines, junctions);
  bool good_infer_right = Detect(image_right, right_features, right_lines);

  bool good_infer = good_infer_left & good_infer_right;
  if(!good_infer){
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer; 
}

BboxPtr FeatureDetector::YoloDetect(cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features) {
  BboxPtr bboxs;
  bool good_infer = false;
  good_infer = _yolodetecter->infer(image, bboxs);
  std::cout << "bboxs size: " << bboxs->size() << std::endl;

  if (!good_infer) {
    std::cout << "Failed when detect yolo !" << std::endl;
  }

  // 创建一个新的矩阵来存储不在检测框内的特征点
  Eigen::Matrix<float, 259, Eigen::Dynamic> filtered_features(259, features.cols());

  int filtered_count = 0;

  // 遍历所有特征点
  for (int i = 0; i < features.cols(); ++i) {
    float x = features(1, i);
    float y = features(2, i);

    bool in_bbox = false;

    // 检查特征点是否在任何一个检测框内
    for (const auto &bbox : *bboxs) {
      if (x >= bbox.x1 && x <= bbox.x2 && y >= bbox.y1 && y <= bbox.y2) {
        in_bbox = true;

        std::cout << "Feature point (" << x << ", " << y << ") is in bbox (" << bbox.x1 << ", " << bbox.y1 << ", " << bbox.x2 << ", " << bbox.y2 << ")" << std::endl;
        break;
      }
    }

    // 如果特征点不在任何检测框内，则保留
    if (!in_bbox) {
      filtered_features.col(filtered_count++) = features.col(i);
    }
  }

  // 调整 filtered_features 的大小
  filtered_features.conservativeResize(259, filtered_count);
  std::cout << "filtered_features size: " << features.cols() << std::endl;
  // 将过滤后的特征点赋值给输入参数
  features = filtered_features;
  std::cout << "filtered_features size: " << features.cols() << std::endl;
  return bboxs;
}
