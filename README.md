# Guided-Filter-Using-CUDA

这是导向滤波/引导滤波的一种GPU实现，经测试，在i9, RTX4090上运行，包括malloc和memcpy操作，**1080P单帧处理可以达到9.8ms**，可以直接或经过Resize缩放后加入到实时视觉任务中

## 介绍

测试场景为去雾工作，输入测试为RGB图的透射图，形式为灰度图，为了可视化保存为了数值范围为[0, 255]的png格式，在main函数中处理为[0, 1]的float类型；引导图为RGB原图，在main函数中处理为[0, 255]的unsigned char类型；输出为[0, 1]的float类型。

其中，RGB原图在进行Guided Filter之前按照BGR通道顺序，处理成了灰度图，如果有需要，可以自行修改[guidedFilter.cu](./guidedfilter.cu/)中的to_float_point函数。

## 实验结果

项目中给出的input测试结果如下

```
input/1_transmission.png
This Time: 140025 us
input/2_transmission.png
This Time: 9481 us
input/3_transmission.png
This Time: 10538 us
input/4_transmission.png
This Time: 9458 us

Average Time: 9825.67 us
```

PS：第一张处理包含GPU WarmUp过程，故计算平均时长时忽略了该值。

---

This is a GPU implementation of the Guided Filter, using CUDA C/C++. It can **_process a 1080P image in 9.8ms_** on Intel Core i9, Nvidia RTX4090, including malloc and memcpy operations. It can be applied directly or after resizing to real-time visual tasks.

## Description

In the test scenario, the input is the transmission map of an RGB image, which is in the form of a grayscale image with values ranging from 0 to 255, saved as a png format for visualization. In the main function, it is processed as a float ranging from 0 to 1. The guide map is the original RGB image, processed as an unsigned char ranging from 0 to 255 in the main function. The output is a float ranging from 0 to 1.

Note that the RGB image is processed into a grayscale image before the filter is applied, according to the BGR channel order. If necessary, you can modify the "to_float_point" function in "[guidedFilter.cu](./guidedFilter.cu)".

## Results

项目中给出的input测试结果如下

The data below shows the processing time in microseconds for a given input image.

```
input/1_transmission.png
This Time: 140025 us
input/2_transmission.png
This Time: 9481 us
input/3_transmission.png
This Time: 10538 us
input/4_transmission.png
This Time: 9458 us

Average Time: 9825.67 us
```

Note that the processing time of the first image includes the GPU WarmUp process, so it is ignored when calculating the average duration.
