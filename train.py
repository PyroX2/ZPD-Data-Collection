import os
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--n-workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="Path to your dataset root. Should contain folders with train, valid and test splits. ")
    parser.add_argument("--ckpt-path", type=str, default="checkpoints", help="Path to directory where model checkpoints will be saved to.")
    return parser.parse_args()

@torch.no_grad()
def validate(model, dataloader, criterion):
    total_loss = 0
    model.eval()
    for input, target in tqdm(dataloader):       
        input, target = input.to(device), target.to(device)
        output = model(input)
        loss = criterion(output, target)
        total_loss += loss.item()

    mean_loss = total_loss / len(dataloader)
    return mean_loss

def train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs):
    best_val_loss = float("inf")

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

        mean_epoch_loss = epoch_train_loss / len(train_dataloader)
        mean_val_loss = validate(model, val_dataloader, criterion)

        print(f"Epoch: {epoch}, Train loss: {mean_epoch_loss}, Val loss: {mean_val_loss}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, "best.pt"))

        torch.save(model.state_dict(), os.path.join(args.ckpt_path, "last.pt"))


args = parse_args()

def main():
    os.makedirs(args.ckpt_path, exist_ok=True)

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

    n_classes = len(os.listdir(os.path.join(args.dataset_root, "train")))

    print(f"Training on {n_classes} classes")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.to(device)

    train_dataset = ImageFolder(os.path.join(args.dataset_root, "train"), transform=train_transforms)
    val_dataset = ImageFolder(os.path.join(args.dataset_root, "valid"), transform=test_transforms)
    test_dataset = ImageFolder(os.path.join(args.dataset_root, "test"), transform=test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_dataloader, val_dataloader, criterion, optimizer, args.epochs)

    print("Testing final model")
    validate(model, test_dataloader, criterion)


if __name__ == "__main__":
    main()