#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    std::string imagePath = R"(D:\HDU\Machine_Vision\count3\2182.png)";
    std::string outputDir;
    bool checkOnly = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--check") checkOnly = true;
        else if (arg == "--output" && i + 1 < argc) outputDir = argv[++i];
        else if (arg == "--output" || arg.compare(0, 2, "--") == 0) {
            std::cerr << "Usage: count3.exe [image_path] [--check] [--output existing_directory]\n";
            return 1;
        }
        else imagePath = arg;
    }

    cv::Mat original = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (original.empty()) {
        std::cerr << "Cannot read image: " << imagePath << std::endl;
        return 1;
    }

    // 1. 灰度化：只使用亮度信息。
    cv::Mat gray;
    cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);

    // 2. 高斯滤波：减弱背景纹理和细小噪声。
    cv::Mat filtered;
    cv::GaussianBlur(gray, filtered, cv::Size(5, 5), 0);

    // 3. 圆孔比工件表面暗，采用大津法反向二值化，圆孔为白色前景。
    cv::Mat binary;
    const double thresholdValue = cv::threshold(filtered, binary, 0, 255,
        cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 4. 小核闭运算连接细小断口，再开运算清理孤立噪点。
    //    核过大会把相邻圆孔连起来，或破坏细线，因此这里只用 3x3。
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat morphology;
    cv::morphologyEx(binary, morphology, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(morphology, morphology, cv::MORPH_OPEN, kernel);

    // 5. 8 邻域连通域标记。标签 0 是背景，不能计入圆孔数量。
    //    面积是白色像素数；圆孔内部的黑色空洞不作为目标计数。
    cv::Mat labels, stats, centroids;
    const int labelCount = cv::connectedComponentsWithStats(
        morphology, labels, stats, centroids, 8, CV_32S);

    // 触边过滤去掉边缘伪影，小面积过滤去掉噪点。
    // 注意：本例假设圆孔完整位于画面内部，触边的圆孔也会被过滤。
    const int minArea = std::max(100, static_cast<int>(original.total() * 0.0005));
    // 只接受完整的近圆形区域；不使用固定坐标，也不把数量固定为 4。
    const double minCircularity = 0.80;
    const double minFillRatio = 0.65;
    std::vector<double> circularities(labelCount, 0.0);
    std::vector<cv::Point2f> circleCenters(labelCount);
    std::vector<float> circleRadii(labelCount, 0.0f);
    std::vector<int> validLabels;
    int borderCount = 0, noiseCount = 0, shapeCount = 0;
    for (int label = 1; label < labelCount; ++label) {
        const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(label, cv::CC_STAT_TOP);
        const int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (x == 0 || y == 0 || x + w == original.cols || y + h == original.rows) {
            ++borderCount;
            continue;
        }
        if (area < minArea) {
            ++noiseCount;
            continue;
        }
        // 先取该连通域的外轮廓，再计算圆度 4*pi*面积/周长^2。
        // 理想圆为 1；结合宽高比、填充率，排除文字、划痕与不规则暗区。
        const cv::Mat mask = (labels == label);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) continue;
        const auto contour = std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
        const double perimeter = cv::arcLength(*contour, true);
        const double circularity = perimeter > 0 ?
            4.0 * CV_PI * cv::contourArea(*contour) / (perimeter * perimeter) : 0;
        const double aspectRatio = static_cast<double>(w) / h;
        const double fillRatio = static_cast<double>(area) / (static_cast<double>(w) * h);
        if (circularity < minCircularity || aspectRatio < 0.80 || aspectRatio > 1.25 ||
            fillRatio < minFillRatio) {
            ++shapeCount;
            continue;
        }
        circularities[label] = circularity;
        cv::minEnclosingCircle(*contour, circleCenters[label], circleRadii[label]);
        validLabels.push_back(label);
    }
    // 按外接框顶部从上到下编号，同高度时从左到右，方便逐一核对。
    std::sort(validLabels.begin(), validLabels.end(), [&](int a, int b) {
        const int ya = stats.at<int>(a, cv::CC_STAT_TOP);
        const int yb = stats.at<int>(b, cv::CC_STAT_TOP);
        return ya != yb ? ya < yb : stats.at<int>(a, cv::CC_STAT_LEFT) < stats.at<int>(b, cv::CC_STAT_LEFT);
    });

    cv::Mat colored = cv::Mat::zeros(original.size(), CV_8UC3);
    cv::Mat acceptedMask = cv::Mat::zeros(original.size(), CV_8UC1);
    cv::Mat result = original.clone();
    cv::RNG rng(2181);
    int number = 0;
    for (int label : validLabels) {
        ++number;
        const cv::Scalar color(rng.uniform(70, 256), rng.uniform(70, 256), rng.uniform(70, 256));
        const cv::Mat mask = (labels == label);
        colored.setTo(color, mask);
        acceptedMask.setTo(255, mask);
        const cv::Rect box(stats.at<int>(label, cv::CC_STAT_LEFT),
            stats.at<int>(label, cv::CC_STAT_TOP), stats.at<int>(label, cv::CC_STAT_WIDTH),
            stats.at<int>(label, cv::CC_STAT_HEIGHT));
        // 绿色圆和红色中心十字标记检测到的完整圆孔。
        const cv::Point center(cvRound(circleCenters[label].x), cvRound(circleCenters[label].y));
        cv::circle(result, center, cvRound(circleRadii[label]), cv::Scalar(0, 220, 0), 2, cv::LINE_AA);
        cv::drawMarker(result, center, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 12, 1, cv::LINE_AA);
        const cv::Point textPoint(box.x, std::max(24, box.y - 5));
        cv::putText(result, std::to_string(number), textPoint,
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        std::cout << "Hole " << number << ": area=" << stats.at<int>(label, cv::CC_STAT_AREA)
                  << ", circularity=" << circularities[label]
                  << ", center=" << circleCenters[label] << ", radius=" << circleRadii[label]
                  << ", box=" << box << '\n';
    }
    cv::putText(result, "Complete holes: " + std::to_string(number), cv::Point(20, 32),
        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    std::cout << "Otsu threshold: " << thresholdValue
              << "\nThreshold polarity: inverted (dark holes)"
              << "\nMinimum area: " << minArea
              << "\nForeground components: " << labelCount - 1
              << "\nRemoved border components: " << borderCount
              << "\nRemoved small components: " << noiseCount
              << "\nRemoved non-circular components: " << shapeCount
              << "\nComplete hole count: " << number << std::endl;

    // 六格流程图，便于同时观察每一步结果。
    const std::vector<cv::Mat> stages = {gray, filtered, binary, morphology, colored, result};
    const std::vector<std::string> titles = {"1 Gray", "2 Gaussian 5x5", "3 Otsu inverted",
        "4 Close + Open 3x3", "5 Accepted circular holes", "6 Count = " + std::to_string(number)};
    const int tileWidth = 400;
    const int imageHeight = cvRound(original.rows * (static_cast<double>(tileWidth) / original.cols));
    const int tileHeight = imageHeight + 32;
    cv::Mat overview(tileHeight * 2, tileWidth * 3, CV_8UC3, cv::Scalar(35, 35, 35));
    for (int i = 0; i < 6; ++i) {
        cv::Mat display;
        if (stages[i].channels() == 1) cv::cvtColor(stages[i], display, cv::COLOR_GRAY2BGR);
        else display = stages[i];
        cv::Mat resized;
        cv::resize(display, resized, cv::Size(tileWidth, imageHeight), 0, 0, cv::INTER_AREA);
        const int x = (i % 3) * tileWidth, y = (i / 3) * tileHeight;
        resized.copyTo(overview(cv::Rect(x, y + 32, tileWidth, imageHeight)));
        cv::putText(overview, titles[i], cv::Point(x + 8, y + 23),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    // 可选输出目录必须已存在；平时 F5 不会写入任何图片文件。
    if (!outputDir.empty()) {
        const std::vector<std::string> names = {"01_gray.png", "02_filtered.png", "03_binary.png",
            "04_morphology.png", "05_labels.png", "06_result.png"};
        try {
            for (int i = 0; i < 6; ++i) {
                if (!cv::imwrite(outputDir + "/" + names[i], stages[i])) return 1;
            }
            if (!cv::imwrite(outputDir + "/07_accepted_mask.png", acceptedMask) ||
                !cv::imwrite(outputDir + "/08_pipeline.png", overview)) return 1;
        }
        catch (const cv::Exception& e) {
            std::cerr << "Cannot save results: " << e.what() << std::endl;
            return 1;
        }
    }
    if (checkOnly) return 0;

    // 显示前按统一比例缩小，宽和高必须乘同一个系数，避免图像被挤扁。
    // AUTOSIZE 让窗口跟随预览图尺寸，不再把图像硬塞进任意比例的窗口。
    const auto makePreview = [](const cv::Mat& image, int maxWidth, int maxHeight) {
        const double scale = std::min(1.0, std::min(
            static_cast<double>(maxWidth) / image.cols,
            static_cast<double>(maxHeight) / image.rows));
        cv::Mat preview;
        cv::resize(image, preview, cv::Size(cvRound(image.cols * scale),
            cvRound(image.rows * scale)), 0, 0, cv::INTER_AREA);
        return preview;
    };
    const cv::Mat pipelinePreview = makePreview(overview, 900, 640);
    const cv::Mat resultPreview = makePreview(result, 640, 640);
    cv::namedWindow("count3 - Complete holes - pipeline", cv::WINDOW_AUTOSIZE);
    cv::imshow("count3 - Complete holes - pipeline", pipelinePreview);
    cv::moveWindow("count3 - Complete holes - pipeline", 20, 20);
    cv::namedWindow("count3 - Complete holes - result", cv::WINDOW_AUTOSIZE);
    cv::imshow("count3 - Complete holes - result", resultPreview);
    cv::moveWindow("count3 - Complete holes - result", 450, 60);
    cv::waitKey(0);  // 点击图片窗口后，按任意键退出。
    cv::destroyAllWindows();
    return 0;
}
