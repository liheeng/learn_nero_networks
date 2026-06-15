import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=2):
        super(VAE, self).__init__()
        # 编码器：提取特征并输出分布参数
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_fc = nn.Linear(hidden_dim, latent_dim)       # 均值输出层
        self.log_var_fc = nn.Linear(hidden_dim, latent_dim)  # 对数方差输出层
        
        # 解码器：从隐向量重建原始数据
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 将像素值归一化到0-1区间
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_fc(h), self.log_var_fc(h)

    # 重参数化技巧：分离随机性，保证梯度可传播
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # 标准正态随机噪声
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def vae_loss(recon_x, x, mu, log_var):
    # 重建损失：二元交叉熵，衡量像素级重建精度
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL散度：解析计算后验分布与标准正态分布的差异
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 128

    # 数据集加载
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 单轮训练示例
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = vae_loss(recon_batch, data, mu, log_var)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    print(f"平均损失: {total_loss / len(train_loader.dataset):.4f}")