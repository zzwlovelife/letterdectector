#ifndef train_h
#define train_h
#include <io.h>
#include <string>
#include <opencv2\ml.hpp>
#include <iostream>
#include <opencv2\opencv.hpp>
using namespace cv;
using namespace std;
using namespace ml;
const String trainImagePath = "D:\\ASUS\\Documents\\opencv\\opencv_letterdectect\\letterSamples\\";
const String imageSuffix = "\\*.png";
const String testImagePath = "D:\\ASUS\\Documents\\opencv\\opencv_letterdectect\\letterSamples\\H\\4_0.846023_gray_7830_2834_step5_recog_3_H_0.965580_0.816903.png";

//生成一个xml文件，并测试一个例子
void train();
#endif // !train_h

