import torch  # 导入 PyTorch，用来演示 DETR 的张量流；这里没有张量 shape。

images = torch.randn(2, 3, 224, 224)  # 构造两张输入图像，shape = (2, 3, 224, 224)。
feature_map = torch.randn(2, 256, 14, 14)  # 假设 CNN backbone 输出特征图，shape = (2, 256, 14, 14)。
sequence = feature_map.flatten(2).transpose(1, 2)  # 把空间特征图拉平成序列，shape = (2, 196, 256)。
pos_embed = torch.randn(1, 196, 256)  # 定义位置编码，shape = (1, 196, 256)。
object_queries = torch.randn(1, 100, 256)  # 定义 100 个 object query，shape = (1, 100, 256)。
encoder_input = sequence + pos_embed  # 给序列加位置编码，shape = (2, 196, 256)。
encoder_output = torch.randn(2, 196, 256)  # 假设 encoder 输出序列，shape = (2, 196, 256)。
decoder_output = torch.randn(2, 100, 256)  # 假设 decoder 输出查询特征，shape = (2, 100, 256)。
class_logits = torch.randn(2, 100, 6)  # 假设检测头输出类别 logits，shape = (2, 100, num_classes_plus_no_object)。
boxes = torch.sigmoid(torch.randn(2, 100, 4))  # 假设检测头输出归一化框坐标，shape = (2, 100, 4)。

print(\"Image shape:\", images.shape)  # 打印输入图像 shape。
print(\"Sequence shape:\", sequence.shape)  # 打印序列化后的特征 shape。
print(\"Encoder output shape:\", encoder_output.shape)  # 打印 encoder 输出 shape。
print(\"Decoder output shape:\", decoder_output.shape)  # 打印 decoder 输出 shape。
print(\"Class logits shape:\", class_logits.shape)  # 打印分类输出 shape。
print(\"Box shape:\", boxes.shape)  # 打印边界框输出 shape。

# å¸¸ç”¨ Metric å·²è¡¥å……ï¼Œä¸‹é¢æ˜¯è¿™é“é¢˜æœ€å¸¸è§æŒ‡æ ‡çš„æœ€å°ç¤ºä¾‹ã€‚
iou_proxy = torch.mean(boxes[..., 2:] - boxes[..., :2])  # ç”¨ä¸€ä¸ªç®€åŒ–ä»£ç†é‡æ¼”ç¤ºæ¡†è´¨é‡ç»Ÿè®¡ï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
print("Box quality proxy:", float(iou_proxy))  # æ‰“å°ä¸€ä¸ªç®€å•çš„æ¡†è´¨é‡ä»£ç†æŒ‡æ ‡ã€‚
