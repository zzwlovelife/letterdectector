#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\imgproc\imgproc.hpp>
#include <io.h>
#include <string>
#include <opencv2\ml.hpp>
using namespace cv;
using namespace std;
using namespace ml;

String trainImagePath = "D:\\ASUS\\Documents\\opencv\\opencv_letterdectect\\letterSamples\\";
String imageSuffix = "\\*.png";
String testImagePath = "D:\\ASUS\\Documents\\opencv\\opencv_letterdectect\\letterSamples\\H\\4_0.846023_gray_7830_2834_step5_recog_3_H_0.965580_0.816903.png";

void part(InputArray src, OutputArray dst)
{
	Mat image;
	Mat imageSource = src.getMat();
	imshow("Source Image", imageSource);
	//cvtColor(imageSource,image, COLOR_BGR2GRAY);	
	blur(imageSource, image, Size(3, 3));
	threshold(image, image, 100, 255, THRESH_BINARY_INV);
	imshow("Threshold Image", image);
	Canny(image, image, 3, 9, 3);
	//threshold(image, image, 0, 255, CV_THRESH_OTSU);
	//寻找最外层轮廓


	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	Mat imageContours = dst.getMat();
	imageContours = Mat::zeros(image.size(), CV_8UC1);	//最小外接矩形画布
   //Mat imageContours1 = Mat::zeros(image.size(), CV_8UC1); //最小外结圆画布
	for (int i = 0; i<int(contours.size()); i++)
	{
		//绘制轮廓
		drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
		//drawContours(imageContours1, contours, i, Scalar(255), 1, 8, hierarchy);


		//绘制轮廓的最小外结矩形
		RotatedRect rect = minAreaRect(contours[i]);
		Point2f P[4];
		rect.points(P);
		for (int j = 0; j <= 3; j++)
		{
			line(imageContours, P[j], P[(j + 1) % 4], Scalar(255), 2);
		}

		//////绘制轮廓的最小外结圆
		//Point2f center; float radius;
		//minEnclosingCircle(contours[i], center, radius);
		//circle(imageContours1, center, int (radius), Scalar(255), 2);

	}
	imshow("MinAreaRect", imageContours);
	//imshow("MinAreaCircle", imageContours1);

	waitKey(0);
	destroyAllWindows();
}
//训练生成xml文件
void train()
{
	////==========================读取图片创建训练数据==============================////
		//将所有图片大小统一转化为8*16
	const int imageRows = 8;
	const int imageCols = 16;
	//图片共有13类
	const int classSum = 13;
	//每类共50张图片
	const int imagesSum = 50;
	//每一行一个训练图片
	float trainingData[classSum * imagesSum][imageRows * imageCols] = { {0} };
	//训练样本标签
	float labels[classSum * imagesSum][classSum] = { {0} };
	Mat src, resizeImg, trainImg;
	for (int i = 0; i < classSum; i++)
	{
		//目标文件夹路径
		//std::string inPath = "E:\\image\\image\\charSamples\\";
		std::string inPath = trainImagePath;
		char temp[256];
		int k = 0;
		sprintf_s(temp, "%c", i + 65);
		cout << temp << endl;
		inPath = inPath + temp + "\\*.png";
		//inPath = inPath + temp + imageSuffix;
		//用于查找的句柄
		intptr_t handle;
		struct _finddata_t fileinfo;
		//第一次查找
		handle = _findfirst(inPath.c_str(), &fileinfo);
		if (handle == -1)
			cout << "error "<< endl;
		do
		{
			//找到的文件的文件名
			std::string imgname = trainImagePath;
			imgname = imgname + temp + "/" + fileinfo.name;
			src = imread(imgname, 0);
			if (src.empty())
			{
				std::cout << "can not load image \n" << std::endl;
				
			}
			//将所有图片大小统一转化为8*16
			resize(src, resizeImg, Size(imageRows, imageCols), (0, 0), (0, 0), INTER_AREA);
			threshold(resizeImg, trainImg, 0, 255, THRESH_BINARY | THRESH_OTSU);
			for (int j = 0; j < imageRows * imageCols; j++)
			{
				trainingData[i * imagesSum + k][j] = (float)resizeImg.data[j];
			}
			// 设置标签数据
			for (int j = 0; j < classSum; j++)
			{
				if (j == i)
					labels[i * imagesSum + k][j] = 1;
				else
					labels[i * imagesSum + k][j] = 0;
			}
			k++;

		} while (!_findnext(handle, &fileinfo));
		Mat labelsMat(classSum * imagesSum, classSum, CV_32FC1, labels);

		_findclose(handle);
	}
	//训练样本数据及对应标签
	Mat trainingDataMat(classSum * imagesSum, imageRows * imageCols, CV_32FC1, trainingData);
	Mat labelsMat(classSum * imagesSum, classSum, CV_32FC1, labels);
	//std::cout<<"trainingDataMat: \n"<<trainingDataMat<<"\n"<<std::endl;
	//std::cout<<"labelsMat: \n"<<labelsMat<<"\n"<<std::endl;
	////==========================训练部分==============================////

	Ptr<ANN_MLP>model = ANN_MLP::create();
	Mat layerSizes = (Mat_<int>(1, 5) << imageRows * imageCols, 128, 128, 128, classSum);
	model->setLayerSizes(layerSizes);
	model->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001));

	Ptr<TrainData> trainData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	model->train(trainData);
	//保存训练结果
	model->save("MLPModel.xml");

	////==========================预测部分==============================////
	//读取测试图像
	Mat test, dst;
	test = imread(testImagePath, 0);
	if (test.empty())
	{
		std::cout << "can not load image \n" << std::endl;

	}
	//将测试图像转化为1*128的向量
	resize(test, test, Size(imageRows, imageCols), (0, 0), (0, 0), INTER_AREA);
	threshold(test, test, 0, 255, THRESH_BINARY | THRESH_OTSU);
	Mat_<float> testMat(1, imageRows * imageCols);
	for (int i = 0; i < imageRows * imageCols; i++)
	{
		testMat.at<float>(0, i) = (float)test.at<uchar>(i / 8, i % 8);
	}
	//使用训练好的MLP model预测测试图像
	model->predict(testMat, dst);
	std::cout << "testMat: \n" << testMat << "\n" << std::endl;
	std::cout << "dst: \n" << dst << "\n" << std::endl;
	double maxVal = 0;
	Point maxLoc;
	minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc);
	char temp[256];
	sprintf_s(temp, "%c", maxLoc.x + 65);
	std::cout << "测试结果：" << temp << "置信度:" << maxVal * 100 << "%" << std::endl;
	imshow("test", test);
	waitKey(0);
}
//测试
void predict(String imagePath)
{
	//将所有图片大小统一转化为8*16
	const int imageRows = 8;
	const int imageCols = 16;
	//读取训练结果
	Ptr<ANN_MLP> model = StatModel::load<ANN_MLP>("MLPModel.xml");
	////==========================预测部分==============================////
	//读取测试图像
	Mat test, dst;
	test = imread(imagePath, 0);
	if (test.empty())
	{
		std::cout << "can not load image \n" << std::endl;
	}
	//将测试图像转化为1*128的向量
	resize(test, test, Size(imageRows, imageCols), (0, 0), (0, 0), INTER_AREA);
	threshold(test, test, 0, 255, THRESH_BINARY | THRESH_OTSU);
	Mat_<float> testMat(1, imageRows * imageCols);
	for (int i = 0; i < imageRows * imageCols; i++)
	{
		testMat.at<float>(0, i) = (float)test.at<uchar>(i / 8, i % 8);
	}
	//使用训练好的MLP model预测测试图像
	model->predict(testMat, dst);
	std::cout << "testMat: \n" << testMat << "\n" << std::endl;
	std::cout << "dst: \n" << dst << "\n" << std::endl;
	double maxVal = 0;
	Point maxLoc;
	minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc);
	char temp[256];
	sprintf_s(temp, "%c", maxLoc.x+65);
	std::cout << "测试结果：" << temp << "置信度:" << maxVal * 100 << "%" << std::endl;
	imshow("test", test);
	waitKey(0);
}
int main()
{
	

	//String filepath = "D:\\ASUS\\Documents\\opencv\\opencv_letterdectect\\sample\\004.png";
	//Mat src = imread(filepath, 0);
	//Mat image;
	//part(src, image);
	//cout << " over" << endl;
	//system("pause");

	predict("D:\\ASUS\\Documents\\opencv\\opencv_letterdectect\\sample\\2019-08-15 141504.png");
		
	return 0;
}