#include "cutPredict.h"

//�ҳ�����ͼ
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
	//Ѱ�����������


	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	Mat imageContours = dst.getMat();
	imageContours = Mat::zeros(image.size(), CV_8UC1);	//��С��Ӿ��λ���
   //Mat imageContours1 = Mat::zeros(image.size(), CV_8UC1); //��С���Բ����
	for (int i = 0; i<int(contours.size()); i++)
	{
		//��������
		drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
		//drawContours(imageContours1, contours, i, Scalar(255), 1, 8, hierarchy);


		//������������С������
		RotatedRect rect = minAreaRect(contours[i]);
		Point2f P[4];
		rect.points(P);
		for (int j = 0; j <= 3; j++)
		{
			line(imageContours, P[j], P[(j + 1) % 4], Scalar(255), 2);
		}

		//////������������С���Բ
		//Point2f center; float radius;
		//minEnclosingCircle(contours[i], center, radius);
		//circle(imageContours1, center, int (radius), Scalar(255), 2);

	}
	imshow("MinAreaRect", imageContours);
	//imshow("MinAreaCircle", imageContours1);

	waitKey(0);
	destroyAllWindows();
}



//����
void predict(InputArray src,char& c,double& value)
{
	//������ͼƬ��Сͳһת��Ϊ8*16
	const int imageRows = 8;
	const int imageCols = 16;
	//��ȡѵ�����
	Ptr<ANN_MLP> model = StatModel::load<ANN_MLP>("MLPModel.xml");
	////==========================Ԥ�ⲿ��==============================////
	//��ȡ����ͼ��
	Mat test, dst;
	cvtColor(src, test, COLOR_BGR2GRAY);
	if (test.empty())
	{
		std::cout << "can not load image \n" << std::endl;
	}
	//������ͼ��ת��Ϊ1*128������
	resize(test, test, Size(imageRows, imageCols), (0, 0), (0, 0), INTER_AREA);
	threshold(test, test, 0, 255, THRESH_BINARY | THRESH_OTSU);
	Mat_<float> testMat(1, imageRows * imageCols);
	for (int i = 0; i < imageRows * imageCols; i++)
	{
		testMat.at<float>(0, i) = (float)test.at<uchar>(i / 8, i % 8);
	}
	//ʹ��ѵ���õ�MLP modelԤ�����ͼ��
	model->predict(testMat, dst);
	std::cout << "testMat: \n" << testMat << "\n" << std::endl;
	std::cout << "dst: \n" << dst << "\n" << std::endl;
	double maxVal = 0;
	Point maxLoc;
	minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc);
	char temp[256];
	sprintf_s(temp, "%c", maxLoc.x + 65);
	c = temp[0];
	value = maxVal * 100;
	std::cout << "���Խ����" << temp << "���Ŷ�:" << value << "%" << std::endl;
	imshow("test", test);
	waitKey(0);
}