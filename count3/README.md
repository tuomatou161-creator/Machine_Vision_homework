# 完整圆孔检测

输入图片：2182.png。当前示例结果：4 个完整圆孔。

灰度化 → 高斯滤波 → 大津法反向二值化 → 形态学处理 → 连通域标记 → 触边与圆形筛选。边缘残缺圆孔不计入。

用 Visual Studio 打开 count3.sln，选择 Debug / x64，点击顶部绿色的本地 Windows 调试器运行。源码为 count3.cpp。

项目依赖本机 D:\OpenCV_venv\props\OpenCV.props；在其他电脑运行时需调整 OpenCV 环境和源码中的图片路径。

