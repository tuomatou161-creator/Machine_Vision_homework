#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
using namespace cv;

int main(int argc, char* argv[])
{
    // R"(...)" 是 C++ 原始字符串，Windows 路径中的反斜杠不需要写两遍。
    Mat srcMat = imread(R"(D:\HDU\Machine_Vision\Lena.jpg)", IMREAD_COLOR);
    if (srcMat.empty()) {
        std::cerr << "Cannot read D:/HDU/Machine_Vision/Lena.jpg" << std::endl;
        return 1;
    }
    Mat original = srcMat.clone();
    int height = srcMat.rows;
    int width = srcMat.cols;

    // OpenCV 的彩色图像通道顺序是 B、G、R。
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            srcMat.at<Vec3b>(j, i)[0] = srcMat.at<Vec3b>(j, i)[0] / 2;
            srcMat.at<Vec3b>(j, i)[1] = srcMat.at<Vec3b>(j, i)[1] / 2;
            srcMat.at<Vec3b>(j, i)[2] = srcMat.at<Vec3b>(j, i)[2] / 2;
        }
    }
    std::cout << "Processed " << width << " x " << height
              << ": each B/G/R channel is divided by 2." << std::endl;
    // --check 仅供命令行验证。平时直接 F5 运行会显示图片。
    if (argc > 1 && std::string(argv[1]) == "--check") return 0;

    imshow("Original", original);
    imshow("Channels divided by 2", srcMat);
    waitKey(0);  // 点击图片窗口后，按任意键退出。
    destroyAllWindows();
    return 0;
}
