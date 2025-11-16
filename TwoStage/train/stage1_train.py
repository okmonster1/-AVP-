import torch
from torch.utils.data import DataLoader
from encoders.contrast_encoder import contrast_encode
from models.contrast_model import ContrastModel
from utils.metrics import evaluate
from utils.dataset import ContrastDataset
from utils.losses import ContrastiveLoss


def run_stage1(device):
    # 数据
    train_data, train_labels = contrast_encode("Datasets/non-AVP/train.txt")
    test_data, test_labels = contrast_encode("Datasets/non-AVP/test.txt")

    train_set = ContrastDataset(train_data, train_labels)
    test_set = ContrastDataset(test_data, test_labels)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=train_set.collate,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=32)

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_contrast = ContrastiveLoss()

    best_acc = 0
    for epoch in range(50):
        model.train()
        for x1, x2, label, y1, y2 in train_loader:
            x1, x2, label, y1, y2 = x1.to(device), x2.to(device), label.to(device), y1.to(device), y2.to(device)
            z1, z2 = model(x1), model(x2)
            loss_contrast = criterion_contrast(z1, z2, label)
            loss_ce = criterion_ce(model.classify(x1), y1) + criterion_ce(model.classify(x2), y2)
            loss = loss_contrast + loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 评估
        acc = evaluate(test_loader, model, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "stage1_best.pth")
            print(f"Epoch {epoch} - Best Acc: {acc:.4f}")