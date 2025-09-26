//
// Created by haoyuefan on 2021/9/22.
//

#ifndef YOLO_DETECTER_H_
#define YOLO_DETECTER_H_

#include <string>
#include <memory>
#include <Eigen/Core>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

#include "3rdparty/tensorrtbuffer/include/buffers.h"
#include "read_configs.h"

using tensorrt_buffer::TensorRTUniquePtr;

class YoloDetecter {
public:
    explicit YoloDetecter(const YoloConfig &yolo_config);

    bool build();

    bool infer(const cv::Mat &image, BboxPtr &bboxes);

    void save_engine();

    bool deserialize_engine();

private:
    int input_width;
    int input_height;
    int resized_width;
    int resized_height;
    float w_scale;
    float h_scale; 

    YoloConfig yolo_config_;

    SuperPointConfig super_point_config_;
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims semi_dims_{};
    nvinfer1::Dims desc_dims_{};
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                           TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                           TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                           TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

    bool process_input(const tensorrt_buffer::BufferManager &buffers, const cv::Mat &image);

    bool process_output(const tensorrt_buffer::BufferManager &buffers, BboxPtr &bboxes);

    BboxPtr postprocess(int img_width, int img_height, float * outs);

};

typedef std::shared_ptr<YoloDetecter> YoloDetecterPtr;

#endif //SUPER_POINT_H_
