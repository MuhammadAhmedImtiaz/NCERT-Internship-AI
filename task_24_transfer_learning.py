# =====================================================================
# PKCERT AI & SOFTWARE DEVELOPMENT INTERNSHIP
# TASK 24: CNNS & TRANSFER LEARNING (CIFAR-10 / RESNET / MOBILENET)
# =====================================================================

import os
import time
import copy
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Reproducibility Config
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sns.set_theme(style="whitegrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# CUSTOM CNN ARCHITECTURE
# =====================================================================
class CustomCIFARNet(nn.Module):
    def __init__(self, drop_rate=0.3):
        super(CustomCIFARNet, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(drop_rate)
        
        # Classifier Head
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x)))) # 32x32 -> 16x16
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_pretrained_model(arch_name, num_classes=10):
    if arch_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif arch_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# =====================================================================
# MAIN EXECUTION GUARD
# =====================================================================
if __name__ == '__main__':
    print(f"PyTorch Execution Device: {device}")

    # Transforms
    cifar_mean, cifar_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    # Datasets
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    full_test  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Subsets for quick execution
    train_sub, _ = random_split(full_train, [2000, 48000], generator=torch.Generator().manual_seed(SEED))
    val_sub, _   = random_split(full_test, [500, 9500], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_sub, batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_sub, batch_size=64, shuffle=False, num_workers=0)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    custom_model = CustomCIFARNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(custom_model.parameters(), lr=0.001)

    print("\n--- Training Custom CNN ---")
    for epoch in range(3):
        custom_model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = custom_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bx.size(0)
        
        custom_model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = custom_model(bx)
                val_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                val_targets.extend(by.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"Epoch {epoch+1:02d}/03 | Val Acc: {val_acc*100:.2f}%")

    # Transfer Learning Evaluation
    print("\n--- Running Transfer Learning ---")
    architectures = ["resnet18", "vgg16", "mobilenet_v2"]
    tl_results = []

    for arch in architectures:
        model = get_pretrained_model(arch).to(device)
        for param in model.parameters(): 
            param.requires_grad = False
            
        if arch == "resnet18":
            for param in model.fc.parameters(): param.requires_grad = True
        elif arch == "vgg16":
            for param in model.classifier[6].parameters(): param.requires_grad = True
        elif arch == "mobilenet_v2":
            for param in model.classifier[1].parameters(): param.requires_grad = True
            
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        
        t0 = time.time()
        for ep in range(1):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                opt.step()
        train_time = time.time() - t0
        
        model.eval()
        t_preds, t_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = torch.argmax(model(bx), dim=1)
                t_preds.extend(preds.cpu().numpy())
                t_targets.extend(by.cpu().numpy())
                
        acc = accuracy_score(t_targets, t_preds)
        tl_results.append({
            "Architecture": arch.upper(),
            "Strategy": "Feature Extraction",
            "Accuracy": f"{acc*100:.2f}%",
            "Train Time": f"{train_time:.2f}s"
        })

    df_tl = pd.DataFrame(tl_results)
    print("\n--- Transfer Learning Results ---")
    print(df_tl.to_string(index=False))

    # Persistence
    torch.save(custom_model.state_dict(), "custom_cifar10_model.pth")
    df_tl.to_csv("transfer_learning_summary.csv", index=False)
    print("\nSaved assets successfully: 'custom_cifar10_model.pth' and 'transfer_learning_summary.csv'!")