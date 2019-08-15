#include "train.h"
//ѵ������xml�ļ�
void train()
{
	////==========================��ȡͼƬ����ѵ������==============================////
		//������ͼƬ��Сͳһת��Ϊ8*16
	const int imageRows = 8;
	const int imageCols = 16;
	//ͼƬ����13��
	const int classSum = 13;
	//ÿ�๲50��ͼƬ
	const int imagesSum = 50;
	//ÿһ��һ��ѵ��ͼƬ
	float trainingData[classSum * imagesSum][imageRows * imageCols] = { {0} };
	//ѵ��������ǩ
	float labels[classSum * imagesSum][classSum] = { {0} };
	Mat src, resizeImg, trainImg;
	for (int i = 0; i < classSum; i++)
	{
		//Ŀ���ļ���·��
		//std::string inPath = "E:\\image\\image\\charSamples\\";
		std::string inPath = trainImagePath;
		char temp[256];
		int k = 0;
		sprintf_s(temp, "%c", i + 65);
		cout << temp << endl;
		inPath = inPath + temp + "\\*.png";
		//inPath = inPath + temp + imageSuffix;
		//���ڲ��ҵľ��
		intptr_t handle;
		struct _finddata_t fileinfo;
		//��һ�β���
		handle = _findfirst(inPath.c_str(), &fileinfo);
		if (handle == -1)
			cout << "error " << endl;
		do
		{
			//�ҵ����ļ����ļ���
			std::string imgname = trainImagePath;
			imgname = imgname + temp + "/" + fileinfo.name;
			src = imread(imgname, 0);
			if (src.empty())
			{
				std::cout << "can not load image \n" << std::endl;

			}
			//������ͼƬ��Сͳһת��Ϊ8*16
			resize(src, resizeImg, Size(imageRows, imageCols), (0, 0), (0, 0), INTER_AREA);
			threshold(resizeImg, trainImg, 0, 255, THRESH_BINARY | THRESH_OTSU);
			for (int j = 0; j < imageRows * imageCols; j++)
			{
				trainingData[i * imagesSum + k][j] = (float)resizeImg.data[j];
			}
			// ���ñ�ǩ����
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
	//ѵ���������ݼ���Ӧ��ǩ
	Mat trainingDataMat(classSum * imagesSum, imageRows * imageCols, CV_32FC1, trainingData);
	Mat labelsMat(classSum * imagesSum, classSum, CV_32FC1, labels);
	//std::cout<<"trainingDataMat: \n"<<trainingDataMat<<"\n"<<std::endl;
	//std::cout<<"labelsMat: \n"<<labelsMat<<"\n"<<std::endl;
	////==========================ѵ������==============================////

	Ptr<ANN_MLP>model = ANN_MLP::create();
	Mat layerSizes = (Mat_<int>(1, 5) << imageRows * imageCols, 128, 128, 128, classSum);
	model->setLayerSizes(layerSizes);
	model->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001));

	Ptr<TrainData> trainData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	model->train(trainData);
	//����ѵ�����
	model->save("MLPModel.xml");

	////==========================Ԥ�ⲿ��==============================////
	//��ȡ����ͼ��
	Mat test, dst;
	test = imread(testImagePath, 0);
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
	std::cout << "���Խ����" << temp << "���Ŷ�:" << maxVal * 100 << "%" << std::endl;
	imshow("test", test);
	waitKey(0);
}