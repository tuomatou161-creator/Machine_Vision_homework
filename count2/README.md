# 硬币计数

输入图片：2183.png。当前示例结果：10 枚硬币。

灰度化 → 高斯滤波 → 大津法二值化 → 形态学处理 → 连通域标记与过滤 → 编号计数。

用 Visual Studio 打开 count2.sln，选择 Debug / x64，点击顶部绿色的本地 Windows 调试器运行。源码为 count2.cpp。

项目依赖本机 D:\OpenCV_venv\props\OpenCV.props；在其他电脑运行时需调整 OpenCV 环境和源码中的图片路径。

