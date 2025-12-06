import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import time


class WiderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), 
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 256), 
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def train_model(model, trainloader, valloader, train_subset, criterion, optimizer, device, num_epochs=50, model_name="Model"):
    model.to(device)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    epoch_times = [] 

    print(f"\nInício do Treino: {model_name}")
    total_start = time.time()
    
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        model.train()
        running_loss = 0
        correct, total = 0, 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_subset)
        epoch_train_acc = correct / total
        
        epoch_val_loss, epoch_val_acc, _, _ = evaluate(model, valloader, criterion, device)
        
        # Guardar Histórico
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        msg = ""
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), f'{model_name}_best.pth')
            msg = "-> Modelo Guardado!"
        
        duration = time.time() - epoch_start
        epoch_times.append(duration)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[{model_name}] Epoch [{epoch+1}/{num_epochs}] | Time: {duration:.1f}s | "
                  f"Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f} {msg}")

    total_time = time.time() - total_start
    print(f"Fim do Treino {model_name}. Tempo Total: {total_time/60:.2f} min.")
    
    stats = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "times": epoch_times,
        "total_time": total_time,
        "best_train_acc": max(train_accs),
        "train_final_loss": train_losses[-1],
        "best_val_acc": max(val_accs),
        "best_val_loss": min(val_losses),
        "val_final_loss": val_losses[-1]
    }
    
    return stats

def setup_data(batch_size=128, validation_split=0.1):
    transform = transforms.Compose([transforms.ToTensor()])

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                         download=True, transform=transform)

    dataset_size = len(full_trainset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.seed(42) 
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_subset = Subset(full_trainset, train_indices)
    val_subset = Subset(full_trainset, val_indices)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader, train_subset, testset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A utilizar dispositivo: {device}")
    
    trainloader, valloader, testloader, train_subset, testset = setup_data()
    criterion = nn.CrossEntropyLoss()

    model_width = WiderCNN()
    params_w = sum(p.numel() for p in model_width.parameters())
    print(f"\nModelo Wider - Parâmetros: {params_w}")
    
    optimizer_width = optim.SGD(model_width.parameters(), lr=0.01, momentum=0.9) 
    
    w_stats = train_model(
        model_width, trainloader, valloader, train_subset, criterion, optimizer_width, device, 
        num_epochs=50, model_name="WiderCNN"
    )

    model_width.load_state_dict(torch.load('WiderCNN_best.pth'))
    
    # Avaliação Final no Test Set (Wider)
    _, test_acc_w, _, _ = evaluate(model_width, testloader, criterion, device)
    w_stats["test_acc"] = test_acc_w


    model_depth = DeeperCNN()
    params_d = sum(p.numel() for p in model_depth.parameters())
    print(f"\nModelo Deeper - Parâmetros: {params_d}")
    
    optimizer_depth = optim.SGD(model_depth.parameters(), lr=0.01, momentum=0.9)
    
    d_stats = train_model(
        model_depth, trainloader, valloader, train_subset, criterion, optimizer_depth, device, 
        num_epochs=50, model_name="DeeperCNN"
    )

    model_depth.load_state_dict(torch.load('DeeperCNN_best.pth'))

    # Avaliação Final no Test Set (Deeper)
    _, test_acc_d, _, _ = evaluate(model_depth, testloader, criterion, device)
    d_stats["test_acc"] = test_acc_d

    epochs = range(1, 51)
    
    plt.figure(figsize=(14, 6))
    
    # Comparação Accuracy Validação
    plt.subplot(1, 2, 1)
    plt.plot(epochs, w_stats["val_accs"], label='Wider', color='blue')
    plt.plot(epochs, d_stats["val_accs"], label='Deeper', color='orange')
    plt.title('Validation Accuracy: Width vs Depth')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparação Loss Validação
    plt.subplot(1, 2, 2)
    plt.plot(epochs, w_stats["val_losses"], label='Wider', color='blue', linestyle='--')
    plt.plot(epochs, d_stats["val_losses"], label='Deeper', color='orange', linestyle='--')
    plt.title('Validation Loss: Width vs Depth')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    header = f"{'Model':<15} | {'Params':<10} | {'Time(m)':<8} | {'Best Train Acc':<15} | {'Train Final Loss':<18} | {'Best Val Acc':<15} | {'Test Acc':<10} | {'Val Best Loss':<15} | {'Val Final Loss':<15}"
    print(header)
    print("-" * len(header))

    print(f"{'WiderCNN':<15} | "
          f"{params_w:<10} | "
          f"{w_stats['total_time']/60:<8.2f} | "
          f"{w_stats['best_train_acc']:<15.4f} | "
          f"{w_stats['train_final_loss']:<18.4f} | "
          f"{w_stats['best_val_acc']:<15.4f} | "
          f"{w_stats['test_acc']:<10.4f} | "
          f"{w_stats['best_val_loss']:<15.4f} | "
          f"{w_stats['val_final_loss']:<15.4f}")

    print(f"{'DeeperCNN':<15} | "
          f"{params_d:<10} | "
          f"{d_stats['total_time']/60:<8.2f} | "
          f"{d_stats['best_train_acc']:<15.4f} | "
          f"{d_stats['train_final_loss']:<18.4f} | "
          f"{d_stats['best_val_acc']:<15.4f} | "
          f"{d_stats['test_acc']:<10.4f} | "
          f"{d_stats['best_val_loss']:<15.4f} | "
          f"{d_stats['val_final_loss']:<15.4f}")
    
    print("="*140)

if __name__ == '__main__':
    main()