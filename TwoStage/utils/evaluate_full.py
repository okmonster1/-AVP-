import torch
from encoders.contrast_encoder import contrast_encode
from encoders.fem_encoder import fem_encode
from models.contrast_model import ContrastModel
from models.transformer_model import TransformerModel
from train.fusion_train import FusionModel
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, matthews_corrcoef

def evaluate_full(device):
    # 加载测试数据
    contrast_data, y_true = contrast_encode("Datasets/non-AVP/test.txt")
    fem_data, _ = fem_encode("Datasets/non-AVP/test.txt")

    # 加载模型
    contrast_model = ContrastModel().to(device)
    transformer_model = TransformerModel().to(device)
    fusion_model = FusionModel().to(device)

    contrast_model.load_state_dict(torch.load("stage1_best.pth", map_location=device))
    transformer_model.load_state_dict(torch.load("stage2_best.pth", map_location=device))
    fusion_model.load_state_dict(torch.load("fusion_best.pth", map_location=device))

    contrast_model.eval()
    transformer_model.eval()
    fusion_model.eval()

    with torch.no_grad():
        z1 = contrast_model(contrast_data.to(device))  # (B, 64)
        x_proj = transformer_model.input_proj(fem_data.to(device))  # (B, 128)
        z2 = transformer_model.fc[:-1](x_proj)  # (B, 64)
        fused = torch.cat([z1, z2], dim=1)  # (B, 128)
        preds = fusion_model(fused).argmax(1).cpu().numpy()
        probs = torch.softmax(fusion_model(fused), dim=1)[:, 1].cpu().numpy()

    return {
        "Accuracy": float(accuracy_score(y_true, preds)),
        "Sensitivity": float(recall_score(y_true, preds)),
        "Specificity": float(recall_score(y_true, preds, pos_label=0)),
        "MCC": float(matthews_corrcoef(y_true, preds)),
        "AUC": float(roc_auc_score(y_true, probs))
    }