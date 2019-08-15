#include "train.h"
#include "cutPredict.h"

int main()
{
	//String filepath = "D:\\ASUS\\Documents\\opencv\\opencv_letterdectect\\sample\\004.png";
	//Mat src = imread(filepath, 0);
	//Mat image;
	//part(src, image);
	//cout << " over" << endl;
	//system("pause");
	Mat test = imread("letterSamples\\K\\6_0.497471_gray_85_60_step5_recog_2_K_0.489183_0.243354.png");
	double value;
	char c ='0';
	predict(test, c ,value);
	cout << c << "  "<<value<<"%"<<endl;
	return 0;
}