字符识别（0-9、A-Z）。比赛只要求实现了A-M的识别
vs2019、opencv4.1
opencv3与opencv2中的神经网络类有很大的不同

charSamples目录下的训练集是从网上下载下来的，10个数字加24个字母，缺少I、O。

letterSamples目录下的训练集是从charSamples中拷贝来的，I是同一张图像复制了50张。

samples目录下是测试例子

MLPModel.xml是该代码训练的特征文件。识别时，需要加载该文件。

