#include "cutPredict.h"

//找出字母方框图，并做识别，_dst为标完矩形的目标图
void part(InputArray src, OutputArray _dst)
{
	Mat image;//处理中间图像
	//获取源图像,需要为灰度图像
	Mat imageSource = src.getMat();
	//创建输出图像
	_dst.create(src.size(), src.type());

	imshow("Source Image", imageSource);

	//cvtColor(imageSource,image, COLOR_BGR2GRAY);
	//均值滤波
	blur(imageSource, image, Size(3, 3));
	//二值化处理
	threshold(image, image, 100, 255, THRESH_BINARY_INV);
	//目标图为二值化处理后的图像，即黑底白字

	imshow("Threshold Image", image);

	//Canny(image, image, 3, 9, 3);//再做边缘检测的效果不佳
	//threshold(image, image, 0, 255, CV_THRESH_OTSU);

	//寻找最外层轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());

	//Mat imageContours = Mat::zeros(image.size(), CV_8UC1);	//最小外接矩形画布
    //Mat imageContours1 = Mat::zeros(image.size(), CV_8UC1); //最小外结圆画布
	for (int i = 0; i<int(contours.size()); i++)
	{
		//绘制轮廓
		//drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
		//drawContours(imageContours1, contours, i, Scalar(255), 1, 8, hierarchy);
		//imshow("轮廓图", imageContours);
		//waitKey(0);

		//最小外接矩形(有旋转角度)
		RotatedRect rect = minAreaRect(contours[i]);
		//最小外接矩形的外接矩形（无角度）
		Rect re = Rect(rect.boundingRect().x-2,rect.boundingRect().y-2, rect.boundingRect().width+4, rect.boundingRect().height+4);
		//将外接矩形裁切下来，保存至image_cut
		Mat image_cut = Mat(image, re).clone();
		//定义字符、相似度
		char c = '0';
		double value;
		imshow("字符", image_cut);
		waitKey(0);
		//将字符做识别
		predict(image_cut, c, value);
		//绘制轮廓的最小外接矩形，在二值图image上绘制线段，颜色为白色
		Point2f P[4];
		rect.points(P);
		for (int j = 0; j <= 3; j++)
		{
			//line(imageContours, P[j], P[(j + 1) % 4], Scalar(255), 2);
			line(image, P[j], P[(j + 1) % 4], Scalar(255), 2);
		}
		//////绘制轮廓的最小外结圆
		//Point2f center; float radius;
		//minEnclosingCircle(contours[i], center, radius);
		//circle(imageContours1, center, int (radius), Scalar(255), 2);
	}
	//将标记后的二值图复制给目标图输出
	image.copyTo(_dst);
	//waitKey(0);
	destroyAllWindows();
}

//测试识别字符，输入为裁切后的字符图，输出为字符与置信度
void predict(InputArray src,char& c,double& value)
{
	//将所有图片大小统一转化为8*16
	const int imageRows = 8;
	const int imageCols = 16;
	//读取训练结果
	Ptr<ANN_MLP> model = StatModel::load<ANN_MLP>("MLPModel.xml");
	////==========================预测部分==============================////
	//读取测试图像
	Mat test, dst;
	test = src.getMat();
	//已经是二值图，不需要再转成灰度图
//	cvtColor(src, test, COLOR_BGR2GRAY);
	if (test.empty())
	{
		std::cout << "can not load image \n" << std::endl;
	}
	//将测试图像转化为1*128的向量
	resize(test, test, Size(imageRows, imageCols), (0, 0), (0, 0), INTER_AREA);
	//threshold(test, test, 0, 255, THRESH_BINARY | THRESH_OTSU);
	Mat_<float> testMat(1, imageRows * imageCols);
	for (int i = 0; i < imageRows * imageCols; i++)
	{
		testMat.at<float>(0, i) = (float)test.at<uchar>(i / 8, i % 8);
	}
	//使用训练好的MLP model预测测试图像
	model->predict(testMat, dst);
	//std::cout << "testMat: \n" << testMat << "\n" << std::endl;
	//std::cout << "dst: \n" << dst << "\n" << std::endl;
	double maxVal = 0;
	Point maxLoc;
	minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc);
	char temp[256];
	sprintf_s(temp, "%c", maxLoc.x + 65);
	c = temp[0];
	value = maxVal * 100;
	std::cout << "测试结果：" << temp << "置信度:" << value << "%" << std::endl;
	imshow("test", test);
	waitKey(0);
}