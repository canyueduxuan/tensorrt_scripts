#include "opencv2/opencv.hpp"
#include "crestereo.hpp"
#include "lightstereo.hpp"
#include "yaml-cpp/yaml.h"

void warm_up()
{
    std::cout << "-------------warm up start---------------" << std::endl;
    YAML::Node config = YAML::LoadFile("/root/catkin_ws/sample/config.yaml");
    std::string model_name = config["eval_models"][0]["name"].as<std::string>();
    std::string onnx_path = config["eval_models"][0]["onnx_path"].as<std::string>();
    std::string engine_path = config["eval_models"][0]["engine_path"].as<std::string>();
    const std::string left_img_path = config["left_img_path"].as<std::string>();
    const std::string right_img_path = config["right_img_path"].as<std::string>();
    cv::Mat left_image = cv::imread(left_img_path, cv::IMREAD_COLOR);
    cv::Mat right_image = cv::imread(right_img_path, cv::IMREAD_COLOR);
    {
        Trt_Model *trt_base;
        trt_base = new Crestereo(onnx_path, engine_path, 1);
        for (int i = 0; i < 100;i++)
        {
            cv::Mat input_mat[2] = {left_image.clone(), right_image.clone()};
            cv::Mat disparity = cv::Mat(120, 160, CV_32FC1);
            trt_base->doInference(input_mat, &disparity);
        }
        delete trt_base;
    }
    model_name = config["eval_models"][1]["name"].as<std::string>();
    onnx_path = config["eval_models"][1]["onnx_path"].as<std::string>();
    engine_path = config["eval_models"][1]["engine_path"].as<std::string>();
    {
        Trt_Model *trt_base;
        trt_base = new Lightstereo(onnx_path, engine_path, 1);
        for (int i = 0; i < 100;i++)
        {
            cv::Mat input_mat[2] = {left_image.clone(), right_image.clone()};
            cv::Mat disparity = cv::Mat(256, 512, CV_32FC1);
            trt_base->doInference(input_mat, &disparity);
        }
        delete trt_base;
    }

    std::cout << "-------------warm up end---------------" << std::endl;
}

int main() {
    warm_up();
    YAML::Node config = YAML::LoadFile("/root/catkin_ws/sample/config.yaml");
    for (int i = 0; i < config["eval_models"].size();i++)
    {
        Trt_Model *trt_base;
        const std::string model_name = config["eval_models"][i]["name"].as<std::string>();
        const std::string onnx_path = config["eval_models"][i]["onnx_path"].as<std::string>();
        const std::string engine_path = config["eval_models"][i]["engine_path"].as<std::string>();
        const std::string left_img_path = config["left_img_path"].as<std::string>();
        const std::string right_img_path = config["right_img_path"].as<std::string>();
        cv::Mat left_image = cv::imread(left_img_path, cv::IMREAD_COLOR);
        cv::Mat right_image = cv::imread(right_img_path, cv::IMREAD_COLOR);
        cv::Mat input_mat[200];//最多100个流
        cv::Mat disparity[100];

        int streams_number = config["streams"].as<int>();
        int infer_number = config["infer_number"].as<int>();
        auto start = std::chrono::high_resolution_clock::now();
        if (model_name == "crestereo")
        {
            trt_base = new Crestereo(onnx_path, engine_path, streams_number);
            
            for (int j = 0; j < infer_number / streams_number;j++)
            {
                for (int k = 0; k < streams_number;k ++)
                {
                    input_mat[2 * k] = left_image.clone();
                    input_mat[2 * k + 1] = right_image.clone();
                    disparity[k] = cv::Mat(120, 160, CV_32FC1);
                }

                trt_base->doInference(input_mat, disparity);
            }

            disparity[0].convertTo(disparity[0], CV_8U, 1.0);
            cv::imwrite("/root/catkin_ws/output_" + model_name + ".png", disparity[0]);
        }
        else if(model_name == "lightstereo")
        {
            trt_base = new Lightstereo(onnx_path, engine_path, streams_number);
            
            for (int j = 0; j < infer_number / streams_number;j++)
            {
                for (int k = 0; k < streams_number;k ++)
                {
                    input_mat[2 * k] = left_image.clone();
                    input_mat[2 * k + 1] = right_image.clone();
                    disparity[k] = cv::Mat(256, 512, CV_32FC1);
                }

                trt_base->doInference(input_mat, disparity);
            }

            disparity[0].convertTo(disparity[0], CV_8U, 1.0);
            cv::imwrite("/root/catkin_ws/output_" + model_name + ".png", disparity[0]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "耗时: " << elapsed.count() << " ms" << std::endl;
        delete trt_base;
    }
    


    // // 创建SGBM对象，参数可调
    // int minDisparity = 0;
    // int numDisparities = 64;  // 视差范围
    // int blockSize = 5;  // 匹配块大小
    // int P1 = 8 * 3 * blockSize * blockSize;  // 平滑项的参数
    // int P2 = 32 * 3 * blockSize * blockSize; // 平滑项的参数
    // int disp12MaxDiff = 1;  // 右视差图和左视差图的最大容忍差异
    // int preFilterCap = 63;  // 预滤波的最大值
    // int uniquenessRatio = 10;  // 确定匹配的唯一性
    // int speckleWindowSize = 100;  // 斑点过滤的窗口大小
    // int speckleRange = 32;  // 斑点过滤的范围
    // // 创建SGBM计算对象
    // cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize);
    // sgbm->setP1(P1);
    // sgbm->setP2(P2);
    // sgbm->setDisp12MaxDiff(disp12MaxDiff);
    // sgbm->setPreFilterCap(preFilterCap);
    // sgbm->setUniquenessRatio(uniquenessRatio);
    // sgbm->setSpeckleWindowSize(speckleWindowSize);
    // sgbm->setSpeckleRange(speckleRange);

    // // 计算视差图
    // cv::resize(left_image, left_image, cv::Size(160, 120));
    // cv::resize(right_image, right_image, cv::Size(160, 120));
    // cv::Mat disparity_sgbm;
    // sgbm->compute(left_image, right_image, disparity_sgbm);
    // disparity_sgbm.convertTo(disparity_sgbm, CV_8U, 1.0 / 16.0);
    // cv::imwrite("/root/catkin_ws/outputsgbm.png", disparity_sgbm);

    return 0;
}