# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便同时写英文和中文注释。
"""MNIST Variational Autoencoder training template with PyTorch."""  # 用一句话说明这个脚本的用途。

import os  # 导入 os，用来设置 Matplotlib 的缓存目录和创建结果输出目录。

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))  # 把 Matplotlib 缓存目录设到当前项目下，避免写入用户目录失败。

import matplotlib  # 导入 Matplotlib 主模块，用来切换无界面后端。

matplotlib.use("Agg")  # 使用无界面后端，避免在终端或沙箱环境里弹图时报图形连接错误。

import matplotlib.pyplot as plt  # 导入画图库，用来画训练和测试 loss 曲线。
import torch  # 导入 PyTorch，用来定义模型、张量和训练流程。
from torch import nn, optim  # 导入神经网络模块和优化器模块，分别用于定义 VAE 和更新参数。
from torch.nn import functional as F  # 导入函数式 API，用来写激活函数和 loss。
from torch.utils.data import DataLoader  # 导入 DataLoader，用来按 batch 读取图像数据。
from torchvision import datasets, transforms  # 导入 torchvision 数据集和图像变换工具，MNIST 官方示例常这样写。
from torchvision.utils import save_image  # 导入保存图片工具，用来保存重建图和随机采样图。


# 这份代码参考了 PyTorch 官方 examples 仓库里的 VAE MNIST 示例，并改成更适合学习的详细注释版。
# 官方仓库: https://github.com/pytorch/examples
# 官方 VAE 示例原始文件: https://raw.githubusercontent.com/pytorch/examples/main/vae/main.py
#
# 以后我们每次写模型代码都默认检查这些问题，并尽量直接落实到代码结构里。
# 1. 怎么加速: device 选择、batch size、混合精度、DataLoader 参数、模型复杂度。
# 2. 多模态吗: 当前这个例子是单模态图像任务，不是多模态；如果以后有图像+文本，要另外设计融合结构。
# 3. 用不用 PyTorch: VAE 是深度生成模型，通常直接用 PyTorch 更自然。
# 4. metric 要有哪些: VAE 主要看总 loss、重建 loss、KL loss，也可以额外看生成图和重建图的质量。
# 5. train/test 要严格分开: 训练阶段更新参数，测试阶段只评估、不回传梯度。
# 6. loss 用什么: 当前用 BCE 重建损失 + KL 散度；如果数据不是 [0,1] 图像，也常改成 MSE。
# 7. 参数怎么调: 学习率、epoch、batch size、latent_dim、hidden_dim、优化器。
# 8. 代码里要留出问题清单: 让后续继续扩展时不漏掉加速、指标和任务定义。


class VAE(nn.Module):  # 定义一个继承自 PyTorch 模块的变分自编码器。
    def __init__(self, input_dim, hidden_dim, latent_dim):  # 初始化时传入输入维度、隐藏层维度和潜变量维度。
        super().__init__()  # 调用父类初始化，让这个类具备 Module 的能力。
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 编码器第一层，输入 shape = (batch_size, 784)，输出 shape = (batch_size, hidden_dim)。
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # 生成均值向量 mu，输出 shape = (batch_size, latent_dim)。
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 生成对数方差 logvar，输出 shape = (batch_size, latent_dim)。
        self.fc3 = nn.Linear(latent_dim, hidden_dim)  # 解码器第一层，输入 shape = (batch_size, latent_dim)，输出 shape = (batch_size, hidden_dim)。
        self.fc4 = nn.Linear(hidden_dim, input_dim)  # 解码器输出层，输出 shape = (batch_size, 784)。
        # 如果后面发现模型表达能力不够，可以把这里改成更深的 MLP，或者换成卷积版 VAE。
        # 但模型更大以后，也要同步关注训练时间、显存占用和是否更容易过拟合。

    def encode(self, x):  # 编码器负责把输入图像压缩成潜空间分布参数。
        h1 = F.relu(self.fc1(x))  # 先经过线性层再过 ReLU，输入 shape = (batch_size, 784)，输出 shape = (batch_size, hidden_dim)。
        mu = self.fc_mu(h1)  # 从隐藏表示得到均值向量 mu，shape = (batch_size, latent_dim)。
        logvar = self.fc_logvar(h1)  # 从隐藏表示得到对数方差 logvar，shape = (batch_size, latent_dim)。
        return mu, logvar  # 返回潜变量分布的两个参数，后面重参数化会用到。

    def reparameterize(self, mu, logvar):  # 重参数化技巧，让采样过程仍然可以反向传播。
        std = torch.exp(0.5 * logvar)  # 根据 logvar 还原标准差 std，shape = (batch_size, latent_dim)。
        eps = torch.randn_like(std)  # 采样标准正态噪声 epsilon，shape 和 std 一样，都是 (batch_size, latent_dim)。
        z = mu + eps * std  # 用 z = mu + sigma * epsilon 做可导采样，shape = (batch_size, latent_dim)。
        return z  # 返回采样后的潜变量。

    def decode(self, z):  # 解码器负责把潜变量还原成图像。
        h3 = F.relu(self.fc3(z))  # 先从潜空间映射回隐藏表示，输入 shape = (batch_size, latent_dim)，输出 shape = (batch_size, hidden_dim)。
        recon = torch.sigmoid(self.fc4(h3))  # 最终输出 0 到 1 之间的像素值，shape = (batch_size, 784)。
        return recon  # 返回重建后的扁平图像。

    def forward(self, x):  # 前向传播把编码、采样、解码连起来。
        x_flat = x.view(-1, 784)  # 把原始图像从 (batch_size, 1, 28, 28) 拉平成 (batch_size, 784)。
        mu, logvar = self.encode(x_flat)  # 编码得到潜变量分布参数，两者 shape 都是 (batch_size, latent_dim)。
        z = self.reparameterize(mu, logvar)  # 通过重参数化得到采样潜变量 z，shape = (batch_size, latent_dim)。
        recon = self.decode(z)  # 解码得到重建图像，shape = (batch_size, 784)。
        return recon, mu, logvar  # 返回重建结果和潜变量统计量，loss 会同时用到它们。


def set_seed(seed: int = 42) -> None:  # 固定随机种子，让实验结果更稳定、更方便复现。
    torch.manual_seed(seed)  # 给 CPU 随机数生成器设种子，后面参数初始化和采样更稳定。
    if torch.cuda.is_available():  # 如果当前环境里有 CUDA GPU，就顺手把 GPU 的随机种子也固定住。
        torch.cuda.manual_seed_all(seed)  # 给所有 CUDA 设备设同一个种子，方便多卡或单卡复现实验。


def get_device():  # 自动选择可用设备，把加速入口集中管理。
    if torch.backends.mps.is_available():  # 先检查 Apple Silicon 的 MPS 后端是否可用。
        return torch.device("mps")  # 返回 MPS 设备对象，后面模型和 batch 都会放到这里。
    if torch.cuda.is_available():  # 如果没有 MPS，再检查 NVIDIA 的 CUDA 是否可用。
        return torch.device("cuda")  # 返回 CUDA 设备对象，后面训练会尽量走 GPU 加速。
    return torch.device("cpu")  # 如果都不可用，就回退到 CPU 上运行。


def prepare_data(batch_size):  # 负责下载 MNIST、做张量变换和构建 DataLoader。
    transform = transforms.ToTensor()  # 把 PIL 图片转成张量，并把像素范围从 [0,255] 归一化到 [0,1]，输出单张图 shape = (1, 28, 28)。
    train_dataset = datasets.MNIST(  # 构建训练集对象。
        root="./data",  # 把数据下载或读取到当前项目下的 data 目录，方便管理。
        train=True,  # 选择训练集部分，大约有 60000 张图像。
        download=True,  # 如果本地没有数据，就自动下载。
        transform=transform,  # 对每张图执行 ToTensor 变换。
    )
    test_dataset = datasets.MNIST(  # 构建测试集对象。
        root="./data",  # 测试集也放到同一个 data 目录。
        train=False,  # 选择测试集部分，大约有 10000 张图像。
        download=True,  # 如果本地没有数据，就自动下载。
        transform=transform,  # 对每张图执行同样的张量变换。
    )

    train_loader = DataLoader(  # 构建训练数据加载器。
        train_dataset,  # 数据源是 MNIST 训练集。
        batch_size=batch_size,  # 每个 batch 取 batch_size 张图，所以 batch_x shape 约是 (batch_size, 1, 28, 28)。
        shuffle=True,  # 训练集要打乱顺序，避免每轮看到完全相同的顺序。
    )
    test_loader = DataLoader(  # 构建测试数据加载器。
        test_dataset,  # 数据源是 MNIST 测试集。
        batch_size=batch_size,  # 每个 batch 也取 batch_size 张图，shape 规则和训练时一致。
        shuffle=False,  # 测试集一般不打乱，保证评估稳定。
    )
    return train_loader, test_loader  # 返回训练和测试加载器。


def vae_loss_function(recon_x, x, mu, logvar):  # 定义 VAE 的总损失，由重建损失和 KL 散度两部分组成。
    x_flat = x.view(-1, 784)  # 把真实图像也拉平成 (batch_size, 784)，这样才能和 recon_x 对齐。
    recon_loss = F.binary_cross_entropy(  # 计算二元交叉熵重建损失。
        recon_x,  # 模型输出的重建图像，shape = (batch_size, 784)，每个值都在 [0,1]。
        x_flat,  # 原始图像拉平后的结果，shape = (batch_size, 784)，每个值也在 [0,1]。
        reduction="sum",  # 按官方示例把所有像素和样本直接求和，输出是标量张量，shape = ()。
    )
    kl_loss = -0.5 * torch.sum(  # 计算 KL 散度，约束潜变量分布接近标准正态分布。
        1 + logvar - mu.pow(2) - logvar.exp()  # 这是 VAE 论文和官方示例里常见的 KL 公式实现。
    )
    total_loss = recon_loss + kl_loss  # 总损失等于重建损失加上 KL 损失，输出仍是标量张量，shape = ()。
    return total_loss, recon_loss, kl_loss  # 同时返回三种 loss，方便训练日志更清楚。


def train_one_epoch(model, train_loader, optimizer, device):  # 单独封装一轮训练逻辑，让主函数更清楚。
    model.train()  # 明确切换到训练模式，后面如果加 Dropout 或 BatchNorm 才不会出错。
    total_loss = 0.0  # 记录这一轮总 loss 的和，Python 浮点数，没有 shape 概念。
    total_recon_loss = 0.0  # 记录这一轮重建损失的和，方便单独观察模型是不是在学重建。
    total_kl_loss = 0.0  # 记录这一轮 KL 损失的和，方便观察潜空间约束是否过强或过弱。

    for batch_x, _ in train_loader:  # 从 DataLoader 里一批一批取图像和标签；这里是无监督学习，所以标签暂时不用。
        batch_x = batch_x.to(device)  # 把当前 batch 的图像移动到目标设备上，shape 约是 (batch_size, 1, 28, 28)。

        optimizer.zero_grad()  # 在反向传播前先清空旧梯度，否则不同 batch 的梯度会累加。
        recon_batch, mu, logvar = model(batch_x)  # 前向传播后得到重建结果和潜变量统计量，shape 分别约是 (batch_size, 784)、(batch_size, latent_dim)、(batch_size, latent_dim)。
        loss, recon_loss, kl_loss = vae_loss_function(recon_batch, batch_x, mu, logvar)  # 计算总损失、重建损失和 KL 损失。
        loss.backward()  # 根据当前 batch 的总损失做反向传播，把梯度写进每个参数的 .grad。
        optimizer.step()  # 使用优化器按梯度方向更新参数，让模型朝更小 loss 的方向移动。

        total_loss += loss.item()  # 把当前 batch 的总损失转成 Python 数字并累加起来。
        total_recon_loss += recon_loss.item()  # 把当前 batch 的重建损失累加起来。
        total_kl_loss += kl_loss.item()  # 把当前 batch 的 KL 损失累加起来。

    dataset_size = len(train_loader.dataset)  # 训练集样本总数大约是 60000，用来把 sum loss 换成平均到每个样本的 loss。
    return {  # 返回这一轮训练阶段的平均指标字典。
        "loss": total_loss / dataset_size,  # 平均总 loss，单位是“每个样本平均损失”。
        "recon_loss": total_recon_loss / dataset_size,  # 平均重建损失。
        "kl_loss": total_kl_loss / dataset_size,  # 平均 KL 损失。
    }


def evaluate(model, test_loader, device, epoch, results_dir):  # 单独封装测试逻辑，让 train/test 严格分开。
    model.eval()  # 明确切换到测试模式，保证评估和训练严格分开。
    total_loss = 0.0  # 记录测试集总 loss 的和。
    total_recon_loss = 0.0  # 记录测试集重建损失的和。
    total_kl_loss = 0.0  # 记录测试集 KL 损失的和。

    with torch.no_grad():  # 进入不计算梯度模式，评估时更省内存也更快。
        for batch_index, (batch_x, _) in enumerate(test_loader):  # 从测试加载器逐批取数据，shape 规则和训练时一致。
            batch_x = batch_x.to(device)  # 把当前测试 batch 的图像移动到目标设备上，shape 约是 (batch_size, 1, 28, 28)。
            recon_batch, mu, logvar = model(batch_x)  # 前向传播得到重建结果和潜变量统计量。
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, batch_x, mu, logvar)  # 计算当前测试 batch 的三种损失。

            total_loss += loss.item()  # 把当前 batch 的总损失加入总和。
            total_recon_loss += recon_loss.item()  # 把当前 batch 的重建损失加入总和。
            total_kl_loss += kl_loss.item()  # 把当前 batch 的 KL 损失加入总和。

            if batch_index == 0:  # 只在测试集第一个 batch 上保存重建图，避免每轮保存太多图片。
                n = min(batch_x.size(0), 8)  # 最多取前 8 张图用于可视化。
                original = batch_x[:n]  # 原始图像 shape = (n, 1, 28, 28)。
                reconstructed = recon_batch.view(batch_x.size(0), 1, 28, 28)[:n]  # 把重建结果从 (batch_size, 784) 还原成 (batch_size, 1, 28, 28)，再取前 n 张。
                comparison = torch.cat([original, reconstructed], dim=0)  # 把原图和重建图在第 0 维拼起来，shape = (2n, 1, 28, 28)。
                save_image(  # 把原图和重建图保存到文件里，方便直观看模型效果。
                    comparison.cpu(),  # 先搬回 CPU 再保存图片。
                    os.path.join(results_dir, f"reconstruction_epoch_{epoch}.png"),  # 以 epoch 命名输出文件。
                    nrow=n,  # 每行放 n 张图，所以第一行是原图，第二行是重建图。
                )

    dataset_size = len(test_loader.dataset)  # 测试集样本总数大约是 10000。
    return {  # 返回测试阶段平均指标字典。
        "loss": total_loss / dataset_size,  # 平均总 loss。
        "recon_loss": total_recon_loss / dataset_size,  # 平均重建损失。
        "kl_loss": total_kl_loss / dataset_size,  # 平均 KL 损失。
    }


def save_samples(model, latent_dim, device, results_dir, epoch):  # 从潜空间随机采样，生成新的数字图像。
    model.eval()  # 生成阶段也切到 eval 模式，和测试阶段保持一致。
    with torch.no_grad():  # 随机采样时也不需要梯度，所以要关闭梯度计算。
        z = torch.randn(64, latent_dim).to(device)  # 从标准正态分布采样 64 个潜变量，shape = (64, latent_dim)。
        sample = model.decode(z).cpu()  # 解码得到 64 张图像，shape = (64, 784)，再搬回 CPU。
        sample = sample.view(64, 1, 28, 28)  # 把扁平结果还原成图像，shape = (64, 1, 28, 28)。
        save_image(  # 保存随机生成图。
            sample,  # 要保存的样本图像。
            os.path.join(results_dir, f"sample_epoch_{epoch}.png"),  # 按 epoch 命名，方便比较不同训练阶段的生成效果。
            nrow=8,  # 8x8 网格展示 64 张图。
        )


def plot_curves(train_history, test_history, plot_path):  # 把画图逻辑单独拆出来，主流程更干净。
    epochs = range(1, len(train_history["loss"]) + 1)  # 生成横轴 epoch 序列，长度等于训练轮数。
    plt.figure(figsize=(15, 4))  # 创建一个宽 15、高 4 的画布，让三张图并排显示。

    plt.subplot(1, 3, 1)  # 选择左边第 1 张子图，专门画总 loss 曲线。
    plt.plot(epochs, train_history["loss"], label="Train Total Loss")  # 训练总 loss 列表长度是 num_epochs，没有张量 shape 概念。
    plt.plot(epochs, test_history["loss"], label="Test Total Loss")  # 测试总 loss 列表长度同样是 num_epochs。
    plt.xlabel("Epoch")  # 设置横轴名称为 Epoch。
    plt.ylabel("Loss")  # 设置纵轴名称为 Loss。
    plt.title("Total Loss")  # 设置总 loss 曲线图标题。
    plt.legend()  # 显示图例，区分训练和测试。

    plt.subplot(1, 3, 2)  # 选择中间第 2 张子图，专门画重建损失曲线。
    plt.plot(epochs, train_history["recon_loss"], label="Train Recon Loss")  # 训练重建损失曲线。
    plt.plot(epochs, test_history["recon_loss"], label="Test Recon Loss")  # 测试重建损失曲线。
    plt.xlabel("Epoch")  # 设置横轴名称为 Epoch。
    plt.ylabel("Loss")  # 设置纵轴名称为 Loss。
    plt.title("Reconstruction Loss")  # 设置重建损失图标题。
    plt.legend()  # 显示图例。

    plt.subplot(1, 3, 3)  # 选择右边第 3 张子图，专门画 KL 损失曲线。
    plt.plot(epochs, train_history["kl_loss"], label="Train KL Loss")  # 训练 KL 损失曲线。
    plt.plot(epochs, test_history["kl_loss"], label="Test KL Loss")  # 测试 KL 损失曲线。
    plt.xlabel("Epoch")  # 设置横轴名称为 Epoch。
    plt.ylabel("Loss")  # 设置纵轴名称为 Loss。
    plt.title("KL Loss")  # 设置 KL 损失图标题。
    plt.legend()  # 显示图例。

    plt.tight_layout()  # 自动调整子图间距，避免标题和坐标轴文字重叠。
    plt.savefig(plot_path, dpi=200)  # 把图直接保存成图片文件，避免 plt.show() 依赖图形界面。


def main():  # 主入口函数，后面你继续扩展模型时就改这里和对应子函数。
    config = {
        "seed": 42,  # 随机种子，控制参数初始化和随机采样的稳定性。
        "batch_size": 128,  # 每个 batch 放 128 张图，所以 batch_x shape 通常是 (128, 1, 28, 28)；想加速时常先试着调大它。
        "learning_rate": 1e-3,  # 学习率，控制每次参数更新的步长大小；太大可能震荡，太小可能学得很慢。
        "num_epochs": 5,  # 总训练轮数，示例先训练 5 轮；如果重建图还很差，可以继续增大。
        "input_dim": 784,  # MNIST 单张图像展平后的维度，等于 28 * 28。
        "hidden_dim": 400,  # 隐藏层维度，控制编码器和解码器的中间表示大小。
        "latent_dim": 20,  # 潜变量维度，决定压缩后的潜空间大小；太小可能重建差，太大可能约束不够。
        "results_dir": "vae_results",  # 保存重建图和采样图的输出目录。
        "plot_path": "vae_training_curves.png",  # 训练曲线图片的保存路径。
    }
    # 这份 config 是后面调参最先看的地方。
    # 常见调参顺序:
    # 1. 先看 learning_rate，因为它最容易让训练“完全不学”或“乱跳”。
    # 2. 再看 batch_size，因为它影响速度、显存和梯度稳定性。
    # 3. 再看 num_epochs，因为轮数太少时你可能只是还没训练够。
    # 4. 再看 latent_dim，因为它直接影响压缩能力和生成质量。
    # 5. 如果图像更复杂，最后再考虑更强的卷积版 VAE。

    os.makedirs(config["results_dir"], exist_ok=True)  # 确保结果目录存在，后面保存图片时不会因为目录不存在而报错。
    set_seed(config["seed"])  # 每次运行前先固定随机种子，减少实验波动。
    device = get_device()  # 统一管理 device 选择，是后面做加速时最先检查的点。
    print(f"Using device: {device}")  # 打印当前实际使用的设备，方便确认是否成功切到 GPU/MPS。

    train_loader, test_loader = prepare_data(config["batch_size"])  # 构建 MNIST 训练和测试加载器。
    model = VAE(config["input_dim"], config["hidden_dim"], config["latent_dim"]).to(device)  # 创建 VAE 模型并移动到目标设备上。
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])  # 创建 Adam 优化器，VAE 这类任务通常比 SGD 更容易训稳。

    train_history = {  # 记录训练阶段的历史指标，后面用来画图。
        "loss": [],  # 每一轮训练总 loss 的历史。
        "recon_loss": [],  # 每一轮训练重建损失的历史。
        "kl_loss": [],  # 每一轮训练 KL 损失的历史。
    }
    test_history = {  # 记录测试阶段的历史指标，后面用来画图。
        "loss": [],  # 每一轮测试总 loss 的历史。
        "recon_loss": [],  # 每一轮测试重建损失的历史。
        "kl_loss": [],  # 每一轮测试 KL 损失的历史。
    }

    for epoch in range(1, config["num_epochs"] + 1):  # 外层循环控制训练轮数，epoch 依次是 1 到 num_epochs。
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)  # 训练完一轮后返回训练阶段指标字典。
        test_metrics = evaluate(model, test_loader, device, epoch, config["results_dir"])  # 在完整测试集上评估，并保存重建图。
        save_samples(model, config["latent_dim"], device, config["results_dir"], epoch)  # 每轮结束后从潜空间随机采样，保存生成图。

        train_history["loss"].append(train_metrics["loss"])  # 把当前轮训练总 loss 追加到列表中。
        train_history["recon_loss"].append(train_metrics["recon_loss"])  # 把当前轮训练重建损失追加到列表中。
        train_history["kl_loss"].append(train_metrics["kl_loss"])  # 把当前轮训练 KL 损失追加到列表中。

        test_history["loss"].append(test_metrics["loss"])  # 把当前轮测试总 loss 追加到列表中。
        test_history["recon_loss"].append(test_metrics["recon_loss"])  # 把当前轮测试重建损失追加到列表中。
        test_history["kl_loss"].append(test_metrics["kl_loss"])  # 把当前轮测试 KL 损失追加到列表中。

        print(  # 输出当前轮数和三种训练/测试损失，方便观察收敛情况。
            f"Epoch {epoch}, "
            f"Train Total Loss: {train_metrics['loss']:.4f}, "
            f"Train Recon Loss: {train_metrics['recon_loss']:.4f}, "
            f"Train KL Loss: {train_metrics['kl_loss']:.4f}, "
            f"Test Total Loss: {test_metrics['loss']:.4f}, "
            f"Test Recon Loss: {test_metrics['recon_loss']:.4f}, "
            f"Test KL Loss: {test_metrics['kl_loss']:.4f}"
        )

    plot_curves(train_history, test_history, config["plot_path"])  # 根据历史列表画出训练过程曲线。
    print(f"Saved curve plot to: {config['plot_path']}")  # 打印训练曲线保存路径，方便确认脚本完整执行结束。
    print(f"Saved reconstructions and samples to: {config['results_dir']}")  # 打印图像输出目录，方便查看重建和生成效果。


if __name__ == "__main__":  # 直接运行这个文件时才执行 main，被别的文件导入时不会自动训练。
    main()
