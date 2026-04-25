import torch  # 导入 PyTorch，用来演示 Faster R-CNN 的张量流；这里没有张量 shape。

images = torch.randn(2, 3, 224, 224)  # 构造两张输入图像，shape = (2, 3, 224, 224)。
backbone_features = torch.randn(2, 256, 14, 14)  # 假设 backbone 输出特征图，shape = (2, 256, 14, 14)。
anchors = torch.randn(2, 9 * 14 * 14, 4)  # 假设 RPN 生成锚框集合，shape = (2, num_anchors, 4)。
rpn_objectness = torch.randn(2, 9 * 14 * 14)  # 假设 RPN 输出每个 anchor 的前景分数，shape = (2, num_anchors)。
proposals = torch.randn(2, 100, 4)  # 假设 RPN 选出 100 个候选框，shape = (2, 100, 4)。
roi_features = torch.randn(2, 100, 256)  # 假设 ROI pooling 后得到候选框特征，shape = (2, 100, 256)。
cls_logits = torch.randn(2, 100, 5)  # 假设 ROI head 输出类别 logits，shape = (2, 100, num_classes)。
bbox_deltas = torch.randn(2, 100, 4)  # 假设 ROI head 输出框回归偏移量，shape = (2, 100, 4)。

print(\"Image shape:\", images.shape)  # 打印输入图像 shape。
print(\"Backbone feature shape:\", backbone_features.shape)  # 打印 backbone 特征图 shape。
print(\"Anchor shape:\", anchors.shape)  # 打印 anchor 张量 shape。
print(\"Proposal shape:\", proposals.shape)  # 打印候选框张量 shape。
print(\"ROI feature shape:\", roi_features.shape)  # 打印 ROI 特征张量 shape。
print(\"Class logit shape:\", cls_logits.shape)  # 打印类别 logits shape。
print(\"BBox delta shape:\", bbox_deltas.shape)  # 打印框回归输出 shape。

# å¸¸ç”¨ Metric å·²è¡¥å……ï¼Œä¸‹é¢æ˜¯è¿™é“é¢˜æœ€å¸¸è§æŒ‡æ ‡çš„æœ€å°ç¤ºä¾‹ã€‚
gt_boxes = torch.tensor([[[10.0, 10.0, 50.0, 50.0]]])  # æž„é€ ä¸€ä¸ªçœŸå®žæ¡†å¼ é‡ï¼Œshape = (1, 1, 4)ã€‚
pred_box = torch.tensor([[[12.0, 12.0, 48.0, 52.0]]])  # æž„é€ ä¸€ä¸ªé¢„æµ‹æ¡†å¼ é‡ï¼Œshape = (1, 1, 4)ã€‚
ix1 = torch.maximum(gt_boxes[..., 0], pred_box[..., 0])  # è®¡ç®—äº¤é›†å·¦ä¸Šè§’ xï¼Œshape = (1, 1)ã€‚
iy1 = torch.maximum(gt_boxes[..., 1], pred_box[..., 1])  # è®¡ç®—äº¤é›†å·¦ä¸Šè§’ yï¼Œshape = (1, 1)ã€‚
ix2 = torch.minimum(gt_boxes[..., 2], pred_box[..., 2])  # è®¡ç®—äº¤é›†å³ä¸‹è§’ xï¼Œshape = (1, 1)ã€‚
iy2 = torch.minimum(gt_boxes[..., 3], pred_box[..., 3])  # è®¡ç®—äº¤é›†å³ä¸‹è§’ yï¼Œshape = (1, 1)ã€‚
inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)  # è®¡ç®—äº¤é›†é¢ç§¯ï¼Œshape = (1, 1)ã€‚
area_gt = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])  # è®¡ç®—çœŸå®žæ¡†é¢ç§¯ï¼Œshape = (1, 1)ã€‚
area_pred = (pred_box[..., 2] - pred_box[..., 0]) * (pred_box[..., 3] - pred_box[..., 1])  # è®¡ç®—é¢„æµ‹æ¡†é¢ç§¯ï¼Œshape = (1, 1)ã€‚
iou = inter / (area_gt + area_pred - inter + 1e-7)  # è®¡ç®— IoUï¼Œshape = (1, 1)ã€‚
print("IoU:", iou)  # æ‰“å° IoUã€‚
