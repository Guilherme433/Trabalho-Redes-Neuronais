import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import time

def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, stochastic_depth_prob=0.0):
        super(ResidualBlock, self).__init__()
        self.sd_prob = stochastic_depth_prob

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
        stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
        stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        #Stochastic Depth
        if self.training and self.sd_prob > 0:
            keep_prob = 1 - self.sd_prob
            mask = torch.bernoulli(torch.full((x.shape[0], 1, 1, 1), keep_prob, device=x.device))
            out = out * mask / keep_prob

        out += self.shortcut(x) 
        out = self.relu(out)
        return out

class Stage3ModernCNN(nn.Module):
    def __init__(self, num_classes=10, stochastic_depth_rate=0.2):
        super(Stage3ModernCNN, self).__init__()
        
        self.in_channels = 64
        self.sd_rate = stochastic_depth_rate
        self.block_idx = 0
        self.total_blocks = 4 
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Backbone
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)
        self.layer4 = self._make_layer(512, stride=2)
        
        # Classificador
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, out_channels, stride):
        sd_prob = self.sd_rate * (self.block_idx / self.total_blocks)
        self.block_idx += 1
        
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, stochastic_depth_prob=sd_prob))
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = self.classifier(out)
        return out

def setup_data_stage3(batch_size=128, validation_split=0.1):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                         download=True, transform=transform_test)

    dataset_size = len(full_trainset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_subset = Subset(full_trainset, train_indices)
    
    val_raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    val_subset_clean = Subset(val_raw, val_indices)

    use_pin_memory = torch.cuda.is_available()
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=use_pin_memory)
    valloader = DataLoader(val_subset_clean, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=use_pin_memory)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=use_pin_memory)

    return trainloader, valloader, testloader, testset


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels

def train_modern_model(model, trainloader, valloader, criterion, optimizer, device, num_epochs=50):
    model.to(device)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    epoch_times = []

    print(f"\nInício do Treino (Stage 3) em: {device}")
    total_start = time.time()
    
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        model.train()
        running_loss = 0
        correct, total = 0, 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            # MixUp
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0, device=device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item() 
        
        train_loss = running_loss / len(trainloader.dataset)
        train_acc = correct / total
        
        # Validação
        val_loss, val_acc, _, _ = evaluate(model, valloader, criterion, device)

        saved_msg = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'stage3_best_model.pth')
            saved_msg = "-> Modelo Guardado!"
        # -----------------------------------------------------
        
        duration = time.time() - epoch_start
        epoch_times.append(duration)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Time: {duration:.1f}s | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} {saved_msg}")

    total_time = time.time() - total_start
    print(f"Tempo Total: {total_time/60:.2f} min.")
    
    stats = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "epoch_times": epoch_times,
        "total_time": total_time,
        "best_train_acc": max(train_accs),
        "train_final_loss": train_losses[-1],
        "best_val_acc": max(val_accs),
        "best_val_loss": min(val_losses),
        "val_final_loss": val_losses[-1]
    }
    
    return stats

def plot_results(stats, all_labels, all_preds, testset):
    epochs = range(1, len(stats["train_losses"]) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, stats["train_losses"], label='Train Loss')
    plt.plot(epochs, stats["val_losses"], label='Val Loss')
    plt.title('Stage 3: Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, stats["train_accs"], label='Train Acc')
    plt.plot(epochs, stats["val_accs"], label='Val Acc')
    plt.title('Stage 3: Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=testset.classes, yticklabels=testset.classes)
    plt.title('Confusion Matrix (Stage 3 Best Model)')
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A utilizar dispositivo: {device}")

    trainloader, valloader, testloader, testset = setup_data_stage3()

    model = Stage3ModernCNN(num_classes=10, stochastic_depth_rate=0.2)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parâmetros: {total_params}") 

    # Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0) 

    # 3. Treino
    stats = train_modern_model(
        model, trainloader, valloader, criterion, optimizer, device, num_epochs=50
    )

    model.load_state_dict(torch.load('stage3_best_model.pth'))

    print("Avaliação Final no Test Set (Melhor Modelo):")
    test_loss, test_acc, preds, labels = evaluate(model, testloader, criterion, device)
    stats["test_acc"] = test_acc # Adiciona ao dicionário
    print(f"Stage 3 Test Accuracy: {test_acc:.4f}")
    
    plot_results(stats, labels, preds, testset)
    
    header = f"{'Model':<15} | {'Params':<10} | {'Time(m)':<8} | {'Best Train Acc':<15} | {'Train Final Loss':<18} | {'Best Val Acc':<15} | {'Test Acc':<10} | {'Val Best Loss':<15} | {'Val Final Loss':<15}"
    print(header)
    print("-" * len(header))

    print(f"{'ModernCNN':<15} | "
          f"{total_params:<10} | "
          f"{stats['total_time']/60:<8.2f} | "
          f"{stats['best_train_acc']:<15.4f} | "
          f"{stats['train_final_loss']:<18.4f} | "
          f"{stats['best_val_acc']:<15.4f} | "
          f"{stats['test_acc']:<10.4f} | "
          f"{stats['best_val_loss']:<15.4f} | "
          f"{stats['val_final_loss']:<15.4f}")
    
    print("="*140)

if __name__ == '__main__':
    main()
