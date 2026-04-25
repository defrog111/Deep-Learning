import torch  # 导入 PyTorch，用来演示 MAE 的 patch 和掩码流程；这里没有张量 shape。

images = torch.randn(2, 3, 32, 32)  # 构造两张输入图像，shape = (2, 3, 32, 32)。
patch_embed = torch.nn.Conv2d(3, 32, kernel_size=8, stride=8)  # 定义 patch embedding 层，输出 shape = (B, 32, 4, 4)。
patch_tokens = patch_embed(images).flatten(2).transpose(1, 2)  # 把图像变成 patch token 序列，shape = (2, 16, 32)。
mask = torch.rand(2, 16) > 0.5  # 随机生成 patch 掩码矩阵，shape = (2, 16)。
visible_tokens = patch_tokens[mask].view(2, -1, 32)  # 取出未被 mask 的 token，shape = (2, num_visible_patches, 32)。
encoder_output = visible_tokens  # 在最小示例里把可见 token 当作 encoder 输出，shape = (2, num_visible_patches, 32)。
mask_token = torch.zeros(1, 1, 32)  # 定义 mask token，shape = (1, 1, 32)。
reconstructed_tokens = torch.cat([encoder_output, mask_token.expand(2, 16 - encoder_output.size(1), 32)], dim=1)  # 拼出重建 token 序列，shape = (2, 16, 32)。
reconstruction_head = torch.nn.Linear(32, 3 * 8 * 8)  # 定义重建头，把每个 token 映射回一个 patch，输出 shape = (B, 16, 192)。
reconstructed_patches = reconstruction_head(reconstructed_tokens)  # 得到重建 patch 序列，shape = (2, 16, 192)。

print(\"Patch token shape:\", patch_tokens.shape)  # 打印 patch token 序列的 shape。
print(\"Mask shape:\", mask.shape)  # 打印 mask 矩阵的 shape。
print(\"Visible token shape:\", visible_tokens.shape)  # 打印可见 token 序列的 shape。
print(\"Reconstructed patch shape:\", reconstructed_patches.shape)  # 打印重建 patch 序列的 shape。

# å¸¸ç”¨ Metric å·²è¡¥å……ï¼Œä¸‹é¢æ˜¯è¿™é“é¢˜æœ€å¸¸è§æŒ‡æ ‡çš„æœ€å°ç¤ºä¾‹ã€‚
target_patches = torch.randn_like(reconstructed_patches)  # æž„é€ ç›®æ ‡ patch å¼ é‡ï¼Œshape ä¸Žé‡å»º patch ç›¸åŒã€‚
reconstruction_loss = torch.mean((reconstructed_patches - target_patches) ** 2)  # è®¡ç®— patch é‡å»º MSEï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
print("Patch reconstruction loss:", float(reconstruction_loss))  # æ‰“å° patch é‡å»ºæŸå¤±ã€‚
