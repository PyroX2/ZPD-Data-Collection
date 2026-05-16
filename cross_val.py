import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18, resnet50, resnet101, inception_v3, vit_b_16, convnext_tiny, convnext_small, convnext_base, convnext_large,
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, Inception_V3_Weights, ViT_B_16_Weights, ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights
)
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from dataset import SubsetWrapper
from metrics import MulticlassMetricsCalculator
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from utils import get_transforms
import pandas as pd


SEED = 0

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)

device = "cuda:1" if torch.cuda.is_available() else "cpu"

models = {
    resnet18: ResNet18_Weights.DEFAULT,
    resnet50: ResNet50_Weights.DEFAULT,
    resnet101: ResNet101_Weights.DEFAULT,
    inception_v3: Inception_V3_Weights.DEFAULT,
    vit_b_16: ViT_B_16_Weights.DEFAULT,
    convnext_tiny: ConvNeXt_Tiny_Weights.DEFAULT,
    convnext_small: ConvNeXt_Small_Weights.DEFAULT,
    convnext_base: ConvNeXt_Base_Weights.DEFAULT,
    convnext_large: ConvNeXt_Large_Weights.DEFAULT,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--n-workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--n-classes", type=int, default=7, help="Number of classes to train on.")
    parser.add_argument("--k-folds", type=int, default=10, help="Number of splits for k-fold cross validation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="Path to your dataset root. Should contain folders with train, valid and test splits. ")
    parser.add_argument("--ckpt-path", type=str, default="checkpoints", help="Path to directory where model checkpoints will be saved to.")
    parser.add_argument("--log-dir", type=str, default="runs", help="Tensorboard logs directory")
    return parser.parse_args()

@torch.no_grad()
def validate(model, dataloader, criterion):
    total_loss = 0
    metrics_calculator = MulticlassMetricsCalculator(args.n_classes, device=device)

    model.eval()
    for inputs, target in tqdm(dataloader):       
        inputs, target = inputs.to(device), target.to(device)
        output = model(inputs)
        loss = criterion(output, target)
        total_loss += loss.item()*inputs.shape[0]

        probs = F.softmax(output, dim=-1)
        metrics_calculator.update(probs.detach(), target.detach())

    val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, val_confusion_matrix = metrics_calculator.compute()
    metrics_calculator.reset()

    mean_loss = total_loss / len(dataloader.dataset)
    return mean_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, val_confusion_matrix

def train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs, writer, fold=None, model_name=None):
    best_val_loss = float("inf")

    metrics_calculator = MulticlassMetricsCalculator(args.n_classes, device=device)

    best_val_accuracy = 0.0
    best_val_f1_score = 0.0
    best_val_auprc = 0.0
    best_val_auroc = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0

    for epoch in range(epochs):
        epoch_train_loss = 0

        model.train()
        for inputs, target in tqdm(train_dataloader):
            optimizer.zero_grad()
            
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)

            if model_name == "inception_v3":
                loss1 = criterion(output.logits, target)
                loss2 = criterion(output.aux_logits, target)
                loss = loss1 + 0.4 * loss2
                output = output.logits
            else:
                loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()*inputs.shape[0]
            probs = F.softmax(output, dim=-1)
            metrics_calculator.update(probs.detach(), target.detach())

        train_accuracy, train_f1_score, train_auprc, train_auroc, train_precision, train_recall, train_confusion_matrix = metrics_calculator.compute()
        metrics_calculator.reset()
        print(f"Epoch: {epoch}, Train accuracy: {train_accuracy}, Train f1 score: {train_f1_score}, Train auprc: {train_auprc}, Train auroc: {train_auroc}, Train precision: {train_precision}, Train recall: {train_recall}")

        mean_epoch_loss = epoch_train_loss / len(train_dataloader.dataset)
        
        writer.add_scalar("Loss/Train", mean_epoch_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        writer.add_scalar("F1_Score/Train", train_f1_score, epoch)
        writer.add_scalar("AUPRC/Train", train_auprc, epoch)
        writer.add_scalar("AUROC/Train", train_auroc, epoch)
        writer.add_scalar("Precision/Train", train_precision, epoch)
        writer.add_scalar("Recall/Train", train_recall, epoch)

        mean_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, val_confusion_matrix = validate(model, val_dataloader, criterion)
        print(f"Epoch: {epoch}, Val accuracy: {val_accuracy}, Val f1 score: {val_f1_score}, Val auprc: {val_auprc}, Val auroc: {val_auroc}, Val precision: {val_precision}, Val recall: {val_recall}")

        writer.add_scalar("Loss/Val", mean_val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_accuracy, epoch)
        writer.add_scalar("F1_Score/Val", val_f1_score, epoch)
        writer.add_scalar("AUPRC/Val", val_auprc, epoch)
        writer.add_scalar("AUROC/Val", val_auroc, epoch)
        writer.add_scalar("Precision/Val", val_precision, epoch)
        writer.add_scalar("Recall/Val", val_recall, epoch)

        print(f"Epoch: {epoch}, Train loss: {mean_epoch_loss}, Val loss: {mean_val_loss}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_val_accuracy = val_accuracy
            best_val_auprc = val_auprc
            best_val_auroc = val_auroc
            best_val_f1_score = val_f1_score
            best_val_precision = val_precision
            best_val_recall = val_recall
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(val_confusion_matrix.cpu().numpy(), annot=True, fmt='g', cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Validation Confusion Matrix (Epoch {epoch})")
            writer.add_figure("Confusion_Matrix/Val_Best", fig, epoch)
            plt.close(fig)

    return best_val_loss, best_val_accuracy, best_val_f1_score, best_val_auprc, best_val_auroc, best_val_precision, best_val_recall

def cross_val(model_class, init_weights, dataset, criterion, n_splits=5):
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    val_losses = []
    val_acc = []
    val_auprc = []
    val_auroc = []
    val_f1 = []
    val_prec = []
    val_recall = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(dataset, dataset.targets)):
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"{model_class.__name__}_fold{fold}"))

        train_transform, test_transform = get_transforms(model_class.__name__, seed=SEED)
        
        if model_class.__name__ == "inception_v3":
            model = model_class(weights=init_weights)
            model.fc = nn.Linear(model.fc.in_features, args.n_classes)
        elif model_class.__name__ == "vit_b_16":
            model = model_class(weights=init_weights)
            model.heads.head = nn.Linear(model.heads.head.in_features, args.n_classes)
        elif model_class.__name__.startswith("convnext"):
            model = model_class(weights=init_weights)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, args.n_classes)
        else:
            model = model_class(weights=init_weights)
            model.fc = nn.Linear(model.fc.in_features, args.n_classes)

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        train_subset = Subset(dataset, indices=train_idx)
        val_subset = Subset(dataset, indices=val_idx)

        train_dataset = SubsetWrapper(train_subset, transform=train_transform)
        val_dataset = SubsetWrapper(val_subset, transform=test_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

        best_val_loss, best_val_accuracy, best_val_f1_score, best_val_auprc, best_val_auroc, best_val_precision, best_val_recall = train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=args.epochs, writer=writer, fold=fold, model_name=model_class.__name__)
        val_losses.append(best_val_loss)
        val_acc.append(best_val_accuracy)
        val_auprc.append(best_val_auprc)
        val_auroc.append(best_val_auroc)
        val_f1.append(best_val_f1_score)
        val_prec.append(best_val_precision)
        val_recall.append(best_val_recall)

        writer.close()

    mean_val_loss = sum(val_losses) / len(val_losses)
    mean_val_acc = sum(val_acc) / len(val_acc)
    mean_val_auprc = sum(val_auprc) / len(val_auprc)
    mean_val_auroc = sum(val_auroc) / len(val_auroc)
    mean_val_f1 = sum(val_f1) / len(val_f1)
    mean_val_prec = sum(val_prec) / len(val_prec)
    mean_val_recall = sum(val_recall) / len(val_recall)
    return mean_val_loss, mean_val_acc, mean_val_f1, mean_val_auprc, mean_val_auroc, mean_val_prec, mean_val_recall


args = parse_args()

def main():
    os.makedirs(args.ckpt_path, exist_ok=True)

    print(f"Training on {args.n_classes} classes")

    train_dataset = ImageFolder(os.path.join(args.dataset_root, "train"))

    criterion = nn.CrossEntropyLoss()

    models_metrics = {}

    for model, init_weights in models.items():
        mean_val_loss, mean_val_acc, mean_val_f1, mean_val_auprc, mean_val_auroc, mean_val_prec, mean_val_recall= cross_val(model, init_weights, train_dataset, criterion, args.k_folds)
        print(f"Model: {model.__name__}, Mean Val Loss: {mean_val_loss}, Mean Val Accuracy: {mean_val_acc}, Mean Val F1 Score: {mean_val_f1}, Mean Val AUPRC: {mean_val_auprc}, Mean Val AUROC: {mean_val_auroc}, Mean Val Precision: {mean_val_prec}, Mean Val Recall: {mean_val_recall}")
        models_metrics[model.__name__] = {
            "val_loss": mean_val_loss,
            "val_accuracy": mean_val_acc.item(),
            "val_f1_score": mean_val_f1.item(),
            "val_auprc": mean_val_auprc.item(),
            "val_auroc": mean_val_auroc.item(),
            "val_precision": mean_val_prec.item(),
            "val_recall": mean_val_recall.item()
        }

    final_metrics_df = pd.DataFrame(models_metrics).T
    final_metrics_df.to_csv("cross_val_results.csv", index=True)
    print(final_metrics_df.sort_values(by="val_auprc", ascending=False))


if __name__ == "__main__":
    main()