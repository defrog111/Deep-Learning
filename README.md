# Transformer 图像去噪

- 标签：核心
- Branch：cv-core-transformer-image-denoising

## 题目
用基于 patch 的 Transformer 做图像去噪，输入带噪图像，输出干净图像。

## 你可以怎么讲
1. 先讲输入张量 shape。
2. 再讲模型每一层把 shape 变成什么样。
3. 再讲 loss 怎么定义。
4. 最后讲这个模型适合什么任务，和常见替代方案的区别。

## 文件说明
- main.py：最小可运行示例，逐句中文注释。
- 代码重点是帮助面试讲清楚结构和数据流，不是追求完整训练工程。

## å¸¸ç”¨ Metric
- MSE
- MAE
- PSNR
- SSIM
