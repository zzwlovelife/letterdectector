#include "train.h"
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
			cout << "error " << endl;
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