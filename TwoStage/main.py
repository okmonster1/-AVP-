import torch
import json
import pandas as pd
from train.stage1_train import run_stage1
from train.stage2_train import run_stage2
from train.fusion_train import run_fusion
from utils.evaluate_full import evaluate_full

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running Stage 1: Contrastive Learning...")
    run_stage1(device)

    print("Running Stage 2: FEM Transformer...")
    run_stage2(device)

    print("Running Fusion Training...")
    run_fusion(device)

    print("Final Evaluation on Test Set...")
    metrics = evaluate_full(device)
    with open("final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    df = pd.DataFrame([metrics])
    df.to_csv("final_metrics.csv", index=False)
    print("✅ All done! Check `final_metrics.json/csv`.")

if __name__ == "__main__":
    main()