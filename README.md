# U-Net 图像分割

- 标签：核心
- Branch：cv-core-unet-image-segmentation

## 题目
用 U-Net 做图像分割，输入图像张量，输出每个像素的类别概率图。

## 你可以怎么讲
1. 先讲输入张量 shape。
2. 再讲模型每一层把 shape 变成什么样。
3. 再讲 loss 怎么定义。
4. 最后讲这个模型适合什么任务，和常见替代方案的区别。

## 文件说明
- main.py：最小可运行示例，逐句中文注释。
- 代码重点是帮助面试讲清楚结构和数据流，不是追求完整训练工程。

## å¸¸ç”¨ Metric
- Pixel Accuracy
- IoU
- mIoU
- Dice
- Precision
- Recall
