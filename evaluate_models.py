import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from metrics import MulticlassMetricsCalculator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_ROOT = "dataset"
BATCH_SIZE = 32
N_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = {
    "Baseline": "checkpoints/baseline/best.pt",
    "Feature Extraction": "checkpoints/feature_ext/best.pt",
    "LR Scheduler": "checkpoints/scheduler/best.pt"
}


def load_model(weights_path):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, dataloader):
    metrics_calculator = MulticlassMetricsCalculator(N_CLASSES, device=DEVICE)

    for input, target in tqdm(dataloader, leave=False):
        input, target = input.to(DEVICE), target.to(DEVICE)
        output = model(input)
        probs = F.softmax(output, dim=-1)
        metrics_calculator.update(probs, target)

    acc, f1, auprc, auroc, prec, rec, cm = metrics_calculator.compute()
    return acc.item(), f1.item(), auroc.item(), cm.cpu().numpy()


def main():
    print("Loading Test Dataset")
    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.Resize((224, 224))
    ])

    test_dataset = ImageFolder(os.path.join(DATASET_ROOT, "test"), transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    class_names = test_dataset.classes

    results = {}
    confusion_matrices = {}

    for name, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            print(f"Skipping {name} - file not found at {path}")
            continue

        print(f"Evaluating {name}")
        model = load_model(path)
        acc, f1, auroc, cm = evaluate(model, test_dataloader)

        results[name] = {'Accuracy': acc, 'F1 Score': f1, 'AUROC': auroc}
        confusion_matrices[name] = cm

    for model_name in results.keys():
        metrics = results[model_name]
        cm = confusion_matrices[model_name]

        safe_name = model_name.replace(" ", "_").lower()

        fig, ax = plt.subplots(figsize=(6, 5))
        metrics_names = ['Accuracy', 'F1 Score', 'AUROC']
        values = [metrics[m] for m in metrics_names]

        bars = ax.bar(metrics_names, values, color='#4C72B0', width=0.5)
        ax.bar_label(bars, fmt='%.3f', padding=3)
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name} - Final Test Metrics')
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(f'{safe_name}_metrics.png', dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name} - Confusion Matrix")

        plt.tight_layout()
        plt.savefig(f'{safe_name}_confusion_matrix.png', dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 5))

        # Calculate True Positives, False Positives, False Negatives
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp

        # Epsilon to prevent division by zero
        eps = 1e-7
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_per_class = 2 * (precision * recall) / (precision + recall + eps)

        bars = ax.bar(class_names, f1_per_class, color='#55A868', width=0.5)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        ax.set_ylabel('F1-Score')
        ax.set_title(f'{model_name} - Per-Class Performance')
        ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(f'{safe_name}_per_class.png', dpi=300)
        plt.close()



if __name__ == "__main__":
    main()