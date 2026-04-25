# LSTM 时间序列预测

- 标签：核心
- Branch：`rnn-core-lstm-forecasting`

## 题目
用最小 LSTM 做单变量时间序列预测，演示 hidden state、cell state、回归输出和常见回归指标。

## 代码要求
1. `main.py` 里给最小可运行例子。
2. 每一句代码都带中文注释。
3. 注释里尽量说明 shape 和变量作用。
4. 这道题对应的主流 metric 也要给出来。

## 常用 Metric
- MSE
- RMSE
- MAE
- R2

## 面试讲法
1. 先讲输入序列和输出序列 shape。
2. 再讲 LSTM 比传统 RNN 多了 cell state。
3. 再讲为什么常取最后一个时间步做预测。
4. 最后讲 loss 和 metric。
