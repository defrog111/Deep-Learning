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
