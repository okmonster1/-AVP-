import torch
from torch.utils.data import DataLoader, TensorDataset
from encoders.contrast_encoder import contrast_encode
from encoders.fem_encoder import fem_encode
from models.contrast_model import ContrastModel
from models.transformer_model import TransformerModel
from utils.metrics import evaluate

# ---------- 融合模型定义 ----------
class FusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)

# ---------- 统一入口函数 ----------
def run_fusion(device):
    # 1. 加载数据
    contrast_data, labels = contrast_encode("Datasets/non-AVP/train.txt")
    fem_data, _           = fem_encode("Datasets/non-AVP/train.txt")

    # 2. 加载预训练模型
    contrast_model = ContrastModel().to(device)
    transformer_model = TransformerModel().to(device)
    contrast_model.load_state_dict(
        torch.load("stage1_best.pth", map_location=device))
    transformer_model.load_state_dict(
        torch.load("stage2_best.pth", map_location=device))

    contrast_model.eval()
    transformer_model.eval()

    # 3. 提取特征并融合
    with torch.no_grad():
        z1 = contrast_model(contrast_data.to(device))  # (batch, 64)
        x_proj = transformer_model.input_proj(fem_data.to(device))  # (batch, 128)
        x_proj = transformer_model.fc[:-1](x_proj)  # (batch, 64)
        fused = torch.cat([z1, x_proj], dim=1)  # (batch, 128)

    # 4. 构建融合数据集
    dataset = TensorDataset(fused, labels)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    # 5. 训练融合分类器
    fusion_model = FusionModel().to(device)
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(50):
        fusion_model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(fusion_model(x), y)
            loss.backward()
            optimizer.step()

        # 快速评估（训练集上）
        fusion_model.eval()
        with torch.no_grad():
            preds = fusion_model(fused).argmax(1).cpu()
            acc   = (preds == labels).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            torch.save(fusion_model.state_dict(), "fusion_best.pth")
            print(f"Fusion Epoch {epoch} - Best Acc: {acc:.4f}")

    print("Fusion training finished.")