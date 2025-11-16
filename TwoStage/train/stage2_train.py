import torch
from torch.utils.data import DataLoader, TensorDataset
from encoders.fem_encoder import fem_encode
from models.transformer_model import TransformerModel
from utils.metrics import evaluate

def run_stage2(device):
    # 数据
    X_train, y_train = fem_encode("Datasets/non-AVP/train.txt")
    X_test, y_test = fem_encode("Datasets/non-AVP/test.txt")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(50):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        acc = evaluate(test_loader, model, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "stage2_best.pth")
            print(f"Epoch {epoch} - Best Acc: {acc:.4f}")