import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=256, output_dim=784):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # 像素值归一化到0-1区间
        )

    def forward(self, z):
        return self.net(z)


# 判别器：判断输入样本的真假概率
class Discriminator(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0-1的真假概率
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    batch_size = 128
    lr = 1e-3

    # 初始化模型、损失函数与独立优化器
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()  # 二元交叉熵损失
    opt_g = optim.Adam(generator.parameters(), lr=lr)
    opt_d = optim.Adam(discriminator.parameters(), lr=lr)

    # 数据集加载
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    generator.train()
    discriminator.train()

    for batch_idx, (real_imgs, _) in enumerate(train_loader):
        real_imgs = real_imgs.view(-1, 784).to(device)
        batch_size_cur = real_imgs.size(0)
        
        # 标签：真实样本为1，生成样本为0
        real_labels = torch.ones(batch_size_cur, 1).to(device)
        fake_labels = torch.zeros(batch_size_cur, 1).to(device)

        # ========== 第一步：训练判别器 ==========
        opt_d.zero_grad()
        # 真实样本损失
        real_out = discriminator(real_imgs)
        loss_real = criterion(real_out, real_labels)
        # 生成样本损失
        z = torch.randn(batch_size_cur, latent_dim).to(device)
        fake_imgs = generator(z)
        fake_out = discriminator(fake_imgs.detach())  # 截断梯度，不更新生成器
        loss_fake = criterion(fake_out, fake_labels)
        # 判别器总损失与参数更新
        loss_d = loss_real + loss_fake
        loss_d.backward()
        opt_d.step()

        # ========== 第二步：训练生成器 ==========
        opt_g.zero_grad()
        # 希望判别器将生成样本误判为真实样本
        fake_out = discriminator(fake_imgs)
        loss_g = criterion(fake_out, real_labels)
        loss_g.backward()
        opt_g.step()

    print(f"判别器损失: {loss_d.item():.4f}, 生成器损失: {loss_g.item():.4f}")