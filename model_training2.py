import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from collections import defaultdict

def main():  # 🔧 Posem tot dins la funció main() per compatibilitat amb multiprocessing a Windows
    # ---------- Reproducibilitat ----------
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    # ---------- Transforms ----------
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ---------- Dataset ----------
    data_dir = "data/train"
    dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)
    class_names = dataset.classes
    num_classes = len(class_names)

    def balance_dataset(dataset, max_per_class=500):
        class_counts = defaultdict(int)
        indices = []
        for idx, (_, label) in enumerate(dataset):
            if class_counts[label] < max_per_class:
                indices.append(idx)
                class_counts[label] += 1
        return Subset(dataset, indices)

    # dataset = balance_dataset(dataset, max_per_class=500)

    # ---------- Mostrem una imatge ----------
    img_tensor, label = dataset[0]
    print(f"Image shape: {img_tensor.shape}")
    plt.imshow(img_tensor.squeeze(), cmap="gray")
    plt.title(f"Label: {class_names[label]}")
    plt.axis("off")
    plt.show()

    eval_subset_size = 1000
    indices = random.sample(range(len(dataset)), eval_subset_size)
    train_eval_subset = Subset(dataset, indices)

    # ---------- Validació ----------
    val_dir = "data/validation"
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_eval)

    # ---------- Model ----------
    class ElVostreModel(nn.Module):
        def __init__(self, num_classes):
            super(ElVostreModel, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    # ---------- Hiperparàmetres ----------
    lr = 0.01
    batch_size = 128
    max_total_time = 600

    # ---------- Entrenament ----------
    device = torch.device("cpu")
    model = ElVostreModel(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # ---------- DataLoaders ----------
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    train_eval_loader = DataLoader(train_eval_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ---------- Funcions d’avaluació ----------
    def evaluate(loader, name):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def evaluate_f1(loader, name):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        f1 = f1_score(all_labels, all_preds, average='micro')
        print(f"{name} F1-micro: {f1 * 100:.2f}%")
        return f1

    # ---------- Entrenament amb límit de temps ----------
    start_time = time.time()
    epoch = 0

    while True:
        epoch += 1
        model.train()
        loop = tqdm(dataloader, desc=f"Època {epoch}", leave=False)

        for i, (images, labels) in enumerate(loop):
            if time.time() - start_time > max_total_time - 10:
                print("Temps màxim assolit. Fi de l'entrenament")
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        val_acc = evaluate(val_loader, f"Validació (després de la època {epoch})")

        if time.time() - start_time > max_total_time - 10:
            break

    # ---------- Avaluació final ----------
    train_acc = evaluate(train_eval_loader, "Train (subset)")
    val_acc = evaluate(val_loader, "Validation (final)")
    val_f1 = evaluate_f1(val_loader, "Validation F1 (final)")

    print(f"\nFinal Metrics:")
    print(f"   Train Accuracy: {train_acc * 100:.2f}%")
    print(f"   Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"   Validation F1-micro: {val_f1 * 100:.2f}%")

# 🔧 Afegit per compatibilitat amb Windows
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
