//
// Created by jsz on 4/18/22.
//

#ifndef HUARAY_CAMERA_CAMERA_H
#define HUARAY_CAMERA_CAMERA_H

#include "../camera_include/IMVApi.h"

#include <exception>

#include <opencv2/opencv.hpp>

class Camera {
public:
    Camera();

    ~Camera();

    int init();

    int getImage(cv::Mat &img);

    void cameraStatistic();

private:
    int setProperty(double exposureTime, int width, int height);
    int status = IMV_OK;
    IMV_HANDLE devHandle = NULL;
    void displayDeviceInfo(IMV_DeviceList deviceInfoList)
    {
        IMV_DeviceInfo*	pDevInfo = NULL;
        unsigned int cameraIndex = 0;
        char vendorNameCat[11];
        char cameraNameCat[16];

        // 打印Title行
        // Print title line
        printf("\nIdx Type Vendor     Model      S/N             DeviceUserID    IP Address    \n");
        printf("------------------------------------------------------------------------------\n");

        for (cameraIndex = 0; cameraIndex < deviceInfoList.nDevNum; cameraIndex++)
        {
            pDevInfo = &deviceInfoList.pDevInfo[cameraIndex];
            // 设备列表的相机索引  最大表示字数：3
            // Camera index in device list, display in 3 characters
            printf("%-3d", cameraIndex + 1);

            // 相机的设备类型（GigE，U3V，CL，PCIe）
            // Camera type
            switch (pDevInfo->nCameraType)
            {
                case typeGigeCamera:printf(" GigE");break;
                case typeU3vCamera:printf(" U3V ");break;
                case typeCLCamera:printf(" CL  ");break;
                case typePCIeCamera:printf(" PCIe");break;
                default:printf("     ");break;
            }

            // 制造商信息  最大表示字数：10
            // Camera vendor name, display in 10 characters
            if (strlen(pDevInfo->vendorName) > 10)
            {
                memcpy(vendorNameCat, pDevInfo->vendorName, 7);
                vendorNameCat[7] = '\0';
                strcat(vendorNameCat, "...");
                printf(" %-10.10s", vendorNameCat);
            }
            else
            {
                printf(" %-10.10s", pDevInfo->vendorName);
            }

            // 相机的型号信息 最大表示字数：10
            // Camera model name, display in 10 characters
            printf(" %-10.10s", pDevInfo->modelName);

            // 相机的序列号 最大表示字数：15
            // Camera serial number, display in 15 characters
            printf(" %-15.15s", pDevInfo->serialNumber);

            // 自定义用户ID 最大表示字数：15
            // Camera user id, display in 15 characters
            if (strlen(pDevInfo->cameraName) > 15)
            {
                memcpy(cameraNameCat, pDevInfo->cameraName, 12);
                cameraNameCat[12] = '\0';
                strcat(cameraNameCat, "...");
                printf(" %-15.15s", cameraNameCat);
            }
            else
            {
                printf(" %-15.15s", pDevInfo->cameraName);
            }

            // GigE相机时获取IP地址
            // IP address of GigE camera
            if (pDevInfo->nCameraType == typeGigeCamera)
            {
                printf(" %s", pDevInfo->DeviceSpecificInfo.gigeDeviceInfo.ipAddress);
            }

            printf("\n");
        }

        return;
    }
};


#endif //HUARAY_CAMERA_CAMERA_H