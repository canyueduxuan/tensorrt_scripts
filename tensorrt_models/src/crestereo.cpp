#include "crestereo.hpp"
#include "opencv2/opencv.hpp"
int Crestereo::doInference(void *input,void *output)
{
    cv::Mat* input_mat = static_cast<cv::Mat*>(input);
    cv::Mat* output_mat = static_cast<cv::Mat*>(output);
    //process input
    for (int i = 0;i < this->stream_number_;i++)
    {
        cv::Mat left = input_mat[2 * i];
        cv::Mat right = input_mat[2 * i + 1];
        cv::resize(left, left, cv::Size(this->bindingDims[0].d[3], this->bindingDims[0].d[2]));
        cv::resize(right, right, cv::Size(this->bindingDims[1].d[3], this->bindingDims[1].d[2]));
        cv::cvtColor(left, left, cv::COLOR_BGR2RGB);
        cv::cvtColor(right, right, cv::COLOR_BGR2RGB);
        left.convertTo(left, CV_32F, 1.0); // 转换为浮动型并归一化
        right.convertTo(right, CV_32F, 1.0);
        float* hostDataBuffer1 = static_cast<float*>(this->buffer_managers_[i]->getHostBuffer("left"));
        float* hostDataBuffer2 = static_cast<float*>(this->buffer_managers_[i]->getHostBuffer("right"));
        int channelSize = this->bindingDims[0].d[3] * this->bindingDims[0].d[2];
        std::vector<cv::Mat> leftChannels(3), rightChannels(3);
        cv::split(left, leftChannels);
        cv::split(right, rightChannels);

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
        this->buffer_managers_[i]->copyInputToDeviceAsync(this->streams_[i]);
    }
    //do inference
    for (int i = 0;i < this->stream_number_;i++)
    {
        bool status = this->contexts_[i]->enqueueV2(this->buffer_managers_[i]->getDeviceBindings().data(), this->streams_[i], nullptr);
        if (!status)
        {
            std::cerr << "Failed to enqueue the inference." << std::endl;
            std::exit(0);
        }
    }
    //copy output to host
    for (int i = 0;i < this->stream_number_;i++)
    {
        this->buffer_managers_[i]->copyOutputToHostAsync(this->streams_[i]);
    }
    //Synchronize and copy output
    for (int i = 0;i < this->stream_number_;i++)
    {
        cudaStreamSynchronize(this->streams_[i]);
        memcpy(output_mat[i].data, this->buffer_managers_[i]->getHostBuffer("output"), this->bindingDims[0].d[3] * this->bindingDims[0].d[2] * sizeof(float));
    }
}