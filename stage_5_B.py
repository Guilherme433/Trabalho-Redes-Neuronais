import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn.functional as F

# ====================================================================
# 1. Arquitetura: SE-ResNet-18 (Igual ao Stage 5 Full)
#    (Mantemos a melhoria de arquitetura para isolar a falta do MixUp)
# ====================================================================

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # SE Block presente
        self.se = SEBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Aplicação da atenção
        out = self.se(out) 

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SEResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(SEResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(SEBasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(SEBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(SEBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(SEBasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * SEBasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

# ====================================================================
# 2. Setup e Loop de Treino
# ====================================================================

def setup_data(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Split Validação
    indices = list(range(len(trainset)))
    split = int(np.floor(0.1 * len(trainset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    train_subset = Subset(trainset, train_idx)
    val_dataset_clean = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    val_subset = Subset(val_dataset_clean, val_idx)

    use_pin = torch.cuda.is_available()
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=use_pin)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_pin)

    return trainloader, valloader, testloader

def train_ablation(model, trainloader, valloader, epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    # Mantemos Label Smoothing (Para isolar apenas a falta do MixUp)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_stats = {'loss': [], 'acc': []}
    val_stats = {'loss': [], 'acc': []}
    lrs = []
    
    best_val_acc = 0.0

    print(f"Início Ablation B (SE-ResNet | LabelSmooth ON | NO MixUp) por {epochs} epochs...")
    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # --- Treino NORMAL (Sem MixUp) ---
            outputs = model(inputs)
            loss = criterion(outputs, targets) # Loss normal com LS
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

        t_loss = running_loss / total
        t_acc = correct / total
        train_stats['loss'].append(t_loss)
        train_stats['acc'].append(t_acc)

        # Validação
        model.eval()
        v_loss = 0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                v_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                v_total += targets.size(0)
                v_correct += predicted.eq(targets).sum().item()
        
        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        val_stats['loss'].append(val_loss)
        val_stats['acc'].append(val_acc)

        # Checkpointing
        saved_msg = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'resnet18_ablation_B_best.pth')
            saved_msg = "-> Saved!"

        duration = time.time() - epoch_start
        # Print menos verboso
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Time: {duration:.1f}s | "
                  f"Train Acc: {t_acc:.4f} | Val Acc: {val_acc:.4f} {saved_msg}")

    total_time = time.time() - total_start
    print(f"\nTreino Concluído. Tempo Total: {total_time/60:.2f} min.")
    return train_stats, val_stats, lrs, total_time

# ====================================================================
# 3. Main Execution
# ====================================================================

def main():
    trainloader, valloader, testloader = setup_data(batch_size=128)

    # Instanciar Modelo COM SEBlock (Igual ao Stage 5 Full)
    model = SEResNet18(num_classes=10)
    params = sum(p.numel() for p in model.parameters())
    print(f"SE-ResNet-18 Parameters: {params/1e6:.2f}M")
    
    # Treino (SEM MixUp)
    t_stats, v_stats, lrs, total_time = train_ablation(model, trainloader, valloader, epochs=200)

    # Avaliação Final
    print("\nA carregar melhor modelo para avaliação...")
    model.load_state_dict(torch.load('resnet18_ablation_B_best.pth'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    final_acc = 100 * correct / total
    print(f"\nFinal Test Set Accuracy (Ablation B - No MixUp): {final_acc:.2f}%")
    
    # ====================================================================
    # 5. Plots (ADICIONADO)
    # ====================================================================
    epochs_range = range(1, 201)
    plt.figure(figsize=(15, 5))
    
    # Gráfico 1: Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, t_stats['acc'], label='Train')
    plt.plot(epochs_range, v_stats['acc'], label='Val')
    plt.title('Accuracy (No MixUp)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfico 2: Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, t_stats['loss'], label='Train Loss')
    plt.plot(epochs_range, v_stats['loss'], label='Val Loss')
    plt.title('Loss Curves (No MixUp)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfico 3: Learning Rate
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, lrs, color='green')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # ====================================================================
    # RESUMO FINAL PARA EXCEL/RELATÓRIO
    # ====================================================================
    
    best_val_acc = max(v_stats['acc'])
    best_train_acc = max(t_stats['acc'])
    final_train_loss = t_stats['loss'][-1]
    best_val_loss = min(v_stats['loss'])
    final_val_loss = v_stats['loss'][-1]
    
    print("\n" + "="*140)
    print("ABLATION STUDY B: REMOVER MIXUP (SE-ResNet + LS only)")
    print("="*140)
    header = f"{'Model':<15} | {'Params(M)':<10} | {'Time(m)':<8} | {'Best Train Acc':<15} | {'Train Final Loss':<18} | {'Best Val Acc':<15} | {'Test Acc':<10} | {'Val Best Loss':<15} | {'Val Final Loss':<15}"
    print(header)
    print("-" * len(header))

    print(f"{'No-MixUp':<15} | "
          f"{params/1e6:<10.2f} | "
          f"{total_time/60:<8.2f} | "
          f"{best_train_acc:<15.4f} | "
          f"{final_train_loss:<18.4f} | "
          f"{best_val_acc:<15.4f} | "
          f"{final_acc/100:<10.4f} | "
          f"{best_val_loss:<15.4f} | "
          f"{final_val_loss:<15.4f}")
    
    print("="*140)

if __name__ == '__main__':
    main()