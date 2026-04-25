# GAN 图像生成

- 标签：扩展
- Branch：cv-extra-gan-image-generation

## 题目
用最小 GAN 结构理解生成器和判别器的对抗训练。

## 你可以怎么讲
1. 先讲输入张量 shape。
2. 再讲模型每一层把 shape 变成什么样。
3. 再讲 loss 怎么定义。
4. 最后讲这个模型适合什么任务，和常见替代方案的区别。

## 文件说明
- main.py：最小可运行示例，逐句中文注释。
- 代码重点是帮助面试讲清楚结构和数据流，不是追求完整训练工程。

## å¸¸ç”¨ Metric
- Generator Loss
- Discriminator Loss
- FID
- IS
- LPIPS
