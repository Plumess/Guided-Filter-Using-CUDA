# Guided-Filter-Using-CUDA

This is a GPU implementation of the Guided Filter, using CUDA C/C++. It can process a 1080P image in 9.8ms on Intel Core i9, Nvidia RTX4090, including malloc and memcpy operations. It can be applied directly or after resizing to real-time visual tasks.

## Description

In the test scenario, the input is the transmission map of an RGB image, which is in the form of a grayscale image with values ranging from 0 to 255, saved as a png format for visualization. In the main function, it is processed as a float ranging from 0 to 1. The guide map is the original RGB image, processed as an unsigned char ranging from 0 to 255 in the main function. The output is a float ranging from 0 to 1.

Note that the RGB image is processed into a grayscale image before the filter is applied, according to the BGR channel order. If necessary, you can modify the "to_float_point" function in "[guidedFilter.cu](./guidedFilter.cu)".

## Experimental Data

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
