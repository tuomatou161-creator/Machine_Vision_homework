from pathlib import Path
import sys
import cv2


def main():
    # r"..." 是 Python 原始字符串，路径中的反斜杠不需要写两遍。
    image_path = r"D:\HDU\Machine_Vision\Lena.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片，请检查路径：{image_path}")

    # // 是整数除法；对每个像素的 B、G、R 三个通道都除以 2。
    # 这是 NumPy 的批量写法，效果与 C++ 版本逐像素循环相同。
    result = image // 2
    height, width = image.shape[:2]
    print(f"已处理 {width} × {height} 图像：每个 B/G/R 通道数值除以 2。")
    if "--check" in sys.argv:
        return

    cv2.imshow("Original", image)
    cv2.imshow("Channels divided by 2", result)
    cv2.waitKey(0)  # 点击图片窗口后，按任意键退出。
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
