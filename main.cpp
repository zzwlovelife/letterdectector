#include "train.h"
#include "cutPredict.h"

int main()
{
	Mat test = imread("samples\\IK.png",0);
	Mat des;
	part(test,des);
	imshow("des", des);
	waitKey(0);
	return 0;
}