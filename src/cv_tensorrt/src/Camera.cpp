//
// Created by jsz on 4/18/22.
//

#include "../include/auto_aim/Camera.h"

Camera::Camera()
{
    this->status = IMV_OK;    IMV_HANDLE temp_devHandle = NULL;
    try
    {
        IMV_DeviceList deviceInfoList;
        this->status = IMV_EnumDevices(&deviceInfoList, interfaceTypeAll);
        if (IMV_OK != this->status)
        {
            printf("Enumeration devices failed! ErrorCode[%d]\n", this->status);
            getchar();
            return;
        }

        if (deviceInfoList.nDevNum < 1)
        {
            printf("no camera\n");
            getchar();
            return;
        }
        displayDeviceInfo(deviceInfoList);
        unsigned int cameraIndex = 0;
        this->status = IMV_CreateHandle(&temp_devHandle, modeByIndex, (void *)&cameraIndex);
        if (IMV_OK != this->status)
        {
            throw "Create devHandle failed";
        }
        this->devHandle = temp_devHandle;
    }
    catch (const char *exp)
    {
        printf("Create devHandle failed! ErrorCode[%d]\n", this->status);
    }
}

int Camera::init()
{
    // open camera
    this->status = IMV_Open(this->devHandle);
    if (IMV_OK != this->status)
    {
        printf("Open camera failed! ErrorCode[%d]\n", this->status);
        return this->status;
    }

    // Set feature value

    this->status = setProperty(7000.00, 640, 640);

    // start grabbing
    this->status = IMV_StartGrabbing(this->devHandle);
    if (IMV_OK != this->status)
    {
        printf("Open camera failed! ErrorCode[%d]\n", this->status);
        return this->status;
    }
    return this->status;
}

int Camera::getImage(cv::Mat &img)
{
    IMV_Frame raw_frame;
    this->status = IMV_GetFrame(this->devHandle, &raw_frame, 300);
    if (IMV_OK != this->status)
    {
        printf("Get raw_frame failed! ErrorCode[%d]\n", this->status);
        return this->status;
    }

    // convert raw image to cv::Mat
    cv::Mat buffer;
    buffer.create(raw_frame.frameInfo.height, raw_frame.frameInfo.width, CV_8U);
    memcpy(buffer.data, raw_frame.pData, raw_frame.frameInfo.width * raw_frame.frameInfo.height);
    //printf("%d\n", raw_frame.frameInfo.width * raw_frame.frameInfo.height);
    cv::cvtColor(buffer, buffer, cv::COLOR_BayerRG2RGB);
    buffer.copyTo(img);
    this->status = IMV_ReleaseFrame(this->devHandle, &raw_frame);
    if (IMV_OK != this->status)
    {
        printf("release raw_frame failed! ErrorCode[%d]\n", this->status);
        return this->status;
    }
    return 1;
}

int Camera::setProperty(double exposureTime, int width, int height)
{
    this->status = IMV_SetDoubleFeatureValue(this->devHandle, "ExposureTime", exposureTime);
    if (IMV_OK != this->status)
    {
        printf("Set ExposureTime value failed! ErrorCode[%d]\n", this->status);
        return this->status;
    }

    this->status = IMV_SetIntFeatureValue(devHandle, "Width", width);
    if (IMV_OK != this->status)
    {
        printf("Set Width value failed! ErrorCode[%d]\n", this->status);
        return this->status;
    }
    this->status = IMV_SetIntFeatureValue(devHandle, "Height", height);
    if (IMV_OK != this->status)
    {
        printf("Set Height value failed! ErrorCode[%d]\n", this->status);
        return this->status;
    }
    return IMV_OK;
}

Camera::~Camera()
{
    if (this->devHandle != nullptr)
    {
        IMV_DestroyHandle(this->devHandle);
    }
}
