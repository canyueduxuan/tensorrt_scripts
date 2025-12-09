#include "opencv2/opencv.hpp"
#include "lightstereo.hpp"
int main() {
    const std::string onnx_path = "/root/tensorrt_scripts/models/lightstereo_s_sceneflow_general_opt_256_512.onnx";
    const std::string engine_path = "/root/tensorrt_scripts/models/lightstereo_s_sceneflow_general_opt_256_512.engine";
    Lightstereo lightstereo(onnx_path, engine_path);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1;i++)
{
    cv::Mat left_image = cv::imread("/root/tensorrt_scripts/images/left_row_2.png", cv::IMREAD_COLOR);
    cv::Mat right_image = cv::imread("/root/tensorrt_scripts/images/right_row_2.png", cv::IMREAD_COLOR);
    cv::resize(left_image, left_image, cv::Size(512, 256));
    cv::resize(right_image, right_image, cv::Size(512, 256));
    left_image.convertTo(left_image, CV_32F, 1.0 / 255.0); // 转换为浮动型并归一化
    right_image.convertTo(right_image, CV_32F, 1.0 / 255.0);
    float* hostDataBuffer1 = static_cast<float*>(lightstereo.buffer_manager->getHostBuffer("left_img"));
    float* hostDataBuffer2 = static_cast<float*>(lightstereo.buffer_manager->getHostBuffer("right_img"));
    int channelSize = 256 * 512;

    std::vector<cv::Mat> leftChannels(3), rightChannels(3);
    cv::split(left_image, leftChannels);
    cv::split(right_image, rightChannels);

    // 左图
    for (int c = 0; c < 3; c++)
    {
        memcpy(
            hostDataBuffer1 + c * channelSize,
            leftChannels[c].ptr<float>(0),
            channelSize * sizeof(float)
        );
    }

    // 右图
    for (int c = 0; c < 3; c++)
    {
        memcpy(
            hostDataBuffer2 + c * channelSize,
            rightChannels[c].ptr<float>(0),
            channelSize * sizeof(float)
        );
    }
    lightstereo.buffer_manager->copyInputToDeviceAsync(lightstereo.stream_);
    bool status = lightstereo.context->enqueueV2(lightstereo.buffer_manager->getDeviceBindings().data(), lightstereo.stream_, nullptr);
    if (!status)
    {
        std::cerr << "Failed to enqueue the inference." << std::endl;
        std::exit(-1);
    }
    lightstereo.buffer_manager->copyOutputToHostAsync(lightstereo.stream_);

    cudaStreamSynchronize(lightstereo.stream_);
}
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Total time taken for 1000 loops: " << duration.count() << " milliseconds." << std::endl;

    cv::Mat disparity(256,512,CV_32FC1);
    memcpy(disparity.data, lightstereo.buffer_manager->getHostBuffer("disp_pred"), 256 * 512 * sizeof(float));

    disparity.convertTo(disparity, CV_8U, 255.0 / 32.0);
    cv::applyColorMap(disparity, disparity, cv::COLORMAP_JET);
    cv::imwrite("/root/tensorrt_scripts/output_lightstereo.png", disparity);
    return 0;
}