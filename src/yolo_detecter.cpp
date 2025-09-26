//
// Created by haoyuefan on 2021/9/22.
//
#include "yolo_detecter.h"
#include <utility>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <set>

using namespace tensorrt_log;
using namespace tensorrt_buffer;

YoloDetecter::YoloDetecter(const YoloConfig &yolo_config): resized_width(640), 
        resized_height(640), yolo_config_(yolo_config), engine_(nullptr) {
    setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
    // setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

bool YoloDetecter::build() {
    std::cout << "build yolo model." << std::endl;
    // cudaSetDevice(2);
    if(deserialize_engine()){
        std::cout << "deserialize_engine success." << std::endl;
        return true;
    }
    std::cout << "createInferBuilder." << std::endl;
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        std::cout << "createInferBuilder failed 0." << std::endl;
        return false;
    }
    std::cout << "createInferBuilder." << std::endl;
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) {
        std::cout << "createInferBuilder failed 1." << std::endl;
        return false;
    }
    std::cout << "createInferBuilder." << std::endl;
    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cout << "createInferBuilder failed 2." << std::endl;
        return false;
    }
    std::cout << "createInferBuilder." << std::endl;
    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        std::cout << "createInferBuilder failed 3." << std::endl;
        return false;
    }
    std::cout << "createInferBuilder." << std::endl;
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cout << "createInferBuilder failed 4." << std::endl;
        return false;
    }
    profile->setDimensions(yolo_config_.input_tensor_names[0].c_str(),
                           nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 640, 640));
    profile->setDimensions(yolo_config_.input_tensor_names[0].c_str(),
                           nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 640, 640));
    profile->setDimensions(yolo_config_.input_tensor_names[0].c_str(),
                           nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 640, 640));
    config->addOptimizationProfile(profile);
    std::cout << "createInferBuilder." << std::endl;
    auto constructed = construct_network(builder, network, config, parser);
    if (!constructed) {
        std::cout << "createInferBuilder failed 5." << std::endl;
        return false;
    }
    std::cout << "createInferBuilder." << std::endl;
    auto profile_stream = makeCudaStream();
    if (!profile_stream) {
        std::cout << "createInferBuilder failed 6." << std::endl;
        return false;
    }
    std::cout << "createInferBuilder." << std::endl;
    config->setProfileStream(*profile_stream);
    TensorRTUniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        std::cout << "createInferBuilder failed 7." << std::endl;
        return false;
    }
    std::cout << "createInferBuilder." << std::endl;
    TensorRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime) {
        std::cout << "createInferBuilder failed 8." << std::endl;
        return false;
    }
    std::cout << "createInferBuilder." << std::endl;
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
        std::cout << "createInferBuilder failed 9." << std::endl;
        return false;
    }
    std::cout << "createInferBuilder." << std::endl;
    save_engine();
    ASSERT(network->getNbInputs() == 1);
    input_dims_ = network->getInput(0)->getDimensions();
    ASSERT(input_dims_.nbDims == 4);
    ASSERT(network->getNbOutputs() == 1);
    semi_dims_ = network->getOutput(0)->getDimensions();
    ASSERT(semi_dims_.nbDims == 3);
    // desc_dims_ = network->getOutput(1)->getDimensions();
    // ASSERT(desc_dims_.nbDims == 4);
    std::cout << "build yolo model success." << std::endl;
    return true;
}

bool YoloDetecter::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                                   TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                   TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                   TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
    auto parsed = parser->parseFromFile(yolo_config_.onnx_file.c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }
    // config->setMaxWorkspaceSize(512_MiB);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    enableDLA(builder.get(), config.get(), yolo_config_.dla_core);
    return true;
}


bool YoloDetecter::infer(const cv::Mat &image_, BboxPtr &bboxes) {
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            return false;
        }
    }

    input_height = image_.rows;
    input_width = image_.cols;
    h_scale = (float)input_height / resized_height;
    w_scale = (float)input_width / resized_width;
    cv::Mat image;
    cv::resize(image_, image, cv::Size(resized_width, resized_height));

    assert(engine_->getNbBindings() == 3);

    const int input_index = engine_->getBindingIndex(yolo_config_.input_tensor_names[0].c_str());

    context_->setBindingDimensions(input_index, nvinfer1::Dims4(1, 1, image.rows, image.cols));

    BufferManager buffers(engine_, 0, context_.get());
    
    ASSERT(yolo_config_.input_tensor_names.size() == 1);
    if (!process_input(buffers, image)) {
        return false;
    }

    buffers.copyInputToDevice();

    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        return false;
    }

    buffers.copyOutputToHost();
    if (!process_output(buffers, bboxes)) {
        return false;
    }

    return true;
}

bool YoloDetecter::process_input(const BufferManager &buffers, const cv::Mat &image) {
    input_dims_.d[2] = image.rows;
    input_dims_.d[3] = image.cols;
    semi_dims_.d[1] = image.rows;
    semi_dims_.d[2] = image.cols;
    desc_dims_.d[1] = 256;
    desc_dims_.d[2] = image.rows / 8;
    desc_dims_.d[3] = image.cols / 8;

    auto *host_data_buffer = static_cast<float *>(buffers.getHostBuffer(yolo_config_.input_tensor_names[0]));
    if (host_data_buffer == nullptr) {
        std::cout << "process_input failed." << std::endl;
        return false;
    }
    for(int c = 0; c < 3; ++c) { // Process each channel
        for(int row = 0; row < image.rows; ++row){
            const uchar *ptr = image.ptr(row);
            int row_shift = row * image.cols;
            for (int col = 0; col < image.cols; ++col) {
                host_data_buffer[c * image.rows * image.cols + row_shift + col] = float(ptr[0]) / 255.0;
                ptr++;
            }
        }
    }
    return true;
}

BboxPtr YoloDetecter::postprocess(int img_width, int img_height, float * outs)
{
    BboxPtr bboxes(new std::vector<tensorrt_buffer::BBox>);

    const int num_boxes = 8400;   // 总候选框数
    const int num_classes = 80;   // 类别数
    std::set<int> dyn_objs= {0,2,15,};

    // 模型输出（模拟数据）：(1, 84, 8400)
    // 第 0-3 为坐标，第 4-83 为类别置信度
    std::vector<std::vector<std::vector<float>>> model_output(1, std::vector<std::vector<float>>(84, std::vector<float>(num_boxes)));
    for (int i = 0; i < 84; ++i) {
        for (int j = 0; j < num_boxes; ++j) {
            model_output[0][i][j] = outs[i * num_boxes + j];
        }
    }
    // 存储最终结果
    std::vector<cv::Rect> boxes;          // 边界框
    std::vector<float> confidences;       // 综合置信度
    std::vector<int> class_ids;           // 类别 ID

    // 遍历所有候选框
    for (int i = 0; i < num_boxes; ++i) {
        // 提取坐标
        float x_center = model_output[0][0][i];
        float y_center = model_output[0][1][i];
        float width = model_output[0][2][i];
        float height = model_output[0][3][i];
        // std::cout << "x_center: " << x_center << ", y_center: " << y_center << ", width: " << width << ", height: " << height << std::endl;
        // 还原到像素坐标
        float x_min = std::max(0.0f, x_center - width / 2.0f) * w_scale;
        float y_min = std::max(0.0f, y_center - height / 2.0f) * h_scale;
        float x_max = std::min(static_cast<float>(img_width), x_center + width / 2.0f) * w_scale;
        float y_max = std::min(static_cast<float>(img_height), y_center + height / 2.0f) * h_scale;

        // 找到类别置信度的最大值
        float max_conf = 0.0f;
        int class_id = -1;
        for (int j = 4; j < 84; ++j) {  // 类别分数从第 4 列开始
            if (model_output[0][j][i] > max_conf) {
                max_conf = model_output[0][j][i];
                class_id = j - 4; // 类别索引
            }
        }

        // 置信度筛选
        if (max_conf > 0.5) { // 置信度阈值
            boxes.emplace_back(cv::Rect(cv::Point(x_min, y_min), cv::Point(x_max, y_max)));
            confidences.push_back(max_conf);
            class_ids.push_back(class_id);
        }
    }

    // 非极大值抑制 (NMS)
    float iou_threshold = 0.4;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, iou_threshold, indices);

    // 输出最终结果
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        float confidence = confidences[idx];
        int class_id = class_ids[idx];
        if (dyn_objs.size() > 0) {
            if (dyn_objs.find(class_id) != dyn_objs.end()) {
                tensorrt_buffer::BBox bbox;
                bbox.x1 = box.x;
                bbox.y1 = box.y;
                bbox.x2 = box.x + box.width;
                bbox.y2 = box.y + box.height;
                bboxes->push_back(bbox);
            }
        } else {
            tensorrt_buffer::BBox bbox;
            bbox.x1 = box.x;
            bbox.y1 = box.y;
            bbox.x2 = box.x + box.width;
            bbox.y2 = box.y + box.height;
            bboxes->push_back(bbox);
        }
        
        std::cout << "Class: " << class_id << ", Confidence: " << confidence
                  << ", Box: (" << box.x << ", " << box.y << ", "
                  << box.x + box.width << ", " << box.y + box.height << ")\n";
    }

    return bboxes;
}


bool YoloDetecter::process_output(const BufferManager &buffers, BboxPtr &bboxes) {
    std::cout << "buffers size." << buffers.size(yolo_config_.output_tensor_names[0]) << std::endl;
    auto *output = static_cast<float *>(buffers.getHostBuffer(yolo_config_.output_tensor_names[0]));
    if (output == nullptr) {
        std::cout << "process_output failed." << std::endl;
        return false;
    }
    bboxes = postprocess(640, 640, output);
    return true;
}

void YoloDetecter::save_engine() {
    if (yolo_config_.engine_file.empty()) return;
    if (engine_ != nullptr) {
        nvinfer1::IHostMemory *data = engine_->serialize();
        std::ofstream file(yolo_config_.engine_file, std::ios::binary);
        if (!file) return;
        file.write(reinterpret_cast<const char *>(data->data()), data->size());
    }
}

bool YoloDetecter::deserialize_engine() {
    std::ifstream file(yolo_config_.engine_file.c_str(), std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        char *model_stream = new char[size];
        file.read(model_stream, size);
        file.close();
	nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
        // if (runtime == nullptr) return false;
        if (runtime == nullptr) {
            delete[] model_stream;
            return false;
        }
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
        delete[] model_stream;
        if (engine_ == nullptr) return false;
        return true;
    }
    return false;
}
