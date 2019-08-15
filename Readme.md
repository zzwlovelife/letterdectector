相关说明
	字符识别（0-9、A-Z）。比赛只要求实现了A-M的识别
	vs2019、opencv4.1
	opencv3与opencv2中的神经网络类有很大的不同

	charSamples目录下的训练集是从网上下载下来的，10个数字加24个字母，缺少I、O。

	letterSamples目录下的训练集是从charSamples中拷贝来的，I是同一张图像复制了50张。

	samples目录下是测试例子

	MLPModel.xml是该代码训练的特征文件。识别时，需要加载该文件。

不需要做旋转时的处理步骤：
	1、预处理（平滑、灰度、二值化、边缘检测）
	2、findContours();寻找轮廓
	3、RotatedRect minAreaRect( InputArray points ); 最小外接矩形
	4、Rect boundingRect() const;//返回包含旋转矩形的最小矩形，即字符框
	5、Mat image_cut = Mat(img, rect);
	   Mat image_copy = image_cut.clone();//切割图片
	6、字符识别

2019-08-15
	已经达到的效果：
	输入电子文档截图，识别结果中，I、M不准确。可能的原因是样本不够或者不够典型。接下来还要考虑
	图片的旋转、重新制作训练集训练。