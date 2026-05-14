import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from metrics import MulticlassMetricsCalculator
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns


SEED = 0

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--n-workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--n-classes", type=int, default=7, help="Number of classes to train on.")
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
    for input, target in tqdm(dataloader):       
        input, target = input.to(device), target.to(device)
        output = model(input)
        loss = criterion(output, target)
        total_loss += loss.item()

        probs = F.softmax(output, dim=-1)
        metrics_calculator.update(probs.detach(), target.detach())

    val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, val_confusion_matrix = metrics_calculator.compute()
    metrics_calculator.reset()

    mean_loss = total_loss / len(dataloader)
    return mean_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, val_confusion_matrix

def train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs, writer):
    best_val_loss = float("inf")

    metrics_calculator = MulticlassMetricsCalculator(args.n_classes, device=device)

    for epoch in range(epochs):
        epoch_train_loss = 0

        model.train()
        for input, target in tqdm(train_dataloader):
            optimizer.zero_grad()
            
            input, target = input.to(device), target.to(device)
            output = model(input)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            probs = F.softmax(output, dim=-1)
            metrics_calculator.update(probs.detach(), target.detach())

        train_accuracy, train_f1_score, train_auprc, train_auroc, train_precision, train_recall, train_confusion_matrix = metrics_calculator.compute()
        metrics_calculator.reset()
        print(f"Epoch: {epoch}, Train accuracy: {train_accuracy}, Train f1 score: {train_f1_score}, Train auprc: {train_auprc}, Train auroc: {train_auroc}, Train precision: {train_precision}, Train recall: {train_recall}")

        mean_epoch_loss = epoch_train_loss / len(train_dataloader)
        
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
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, "best.pt"))
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(val_confusion_matrix.cpu().numpy(), annot=True, fmt='g', cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Validation Confusion Matrix (Epoch {epoch})")
            writer.add_figure("Confusion_Matrix/Val_Best", fig, epoch)

        torch.save(model.state_dict(), os.path.join(args.ckpt_path, "last.pt"))


args = parse_args()

def main():
    os.makedirs(args.ckpt_path, exist_ok=True)
    
    writer = SummaryWriter(log_dir=args.log_dir)

    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.Resize((224, 224))
    ])
    
    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.Resize((224, 224))
    ])

    print(f"Training on {args.n_classes} classes")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, args.n_classes)
    model.to(device)

    train_dataset = ImageFolder(os.path.join(args.dataset_root, "train"), transform=train_transforms)
    val_dataset = ImageFolder(os.path.join(args.dataset_root, "valid"), transform=test_transforms)
    test_dataset = ImageFolder(os.path.join(args.dataset_root, "test"), transform=test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_dataloader, val_dataloader, criterion, optimizer, args.epochs, writer)

    print("Testing final model")
    
    model.load_state_dict(torch.load(os.path.join(args.ckpt_path, "best.pt")))
    
    test_loss, test_accuracy, test_f1_score, test_auprc, test_auroc, test_precision, test_recall, test_confusion_matrix = validate(model, test_dataloader, criterion)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1_score:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    writer.add_scalar("Loss/Test", test_loss, args.epochs)
    writer.add_scalar("Accuracy/Test", test_accuracy, args.epochs)
    writer.add_scalar("F1_Score/Test", test_f1_score, args.epochs)
    writer.add_scalar("AUPRC/Test", test_auprc, args.epochs)
    writer.add_scalar("AUROC/Test", test_auroc, args.epochs)
    writer.add_scalar("Precision/Test", test_precision, args.epochs)
    writer.add_scalar("Recall/Test", test_recall, args.epochs)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(test_confusion_matrix.cpu().numpy(), annot=True, fmt='g', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")
    writer.add_figure("Confusion_Matrix/Test", fig, args.epochs)
    
    writer.close()


if __name__ == "__main__":
    main()