#ifndef cutPredict_h
#define cutPredict_h
#include<opencv2\opencv.hpp>
#include<opencv2\ml\ml.hpp>
using namespace cv;
using namespace cv::ml;
using namespace std;
void part(InputArray src, OutputArray dst);
void predict(InputArray src,char& c, double& value);
#endif // !cutPredict_h

