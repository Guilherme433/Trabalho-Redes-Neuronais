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

class BaselineCNN(nn.Module):
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
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def setup_data(batch_size=128, validation_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

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

    #Dataloaders
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader, testset

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
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

def train_one_epoch(model, trainloader, criterion, optimizer, device):
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
        
    return running_loss / total, correct / total

def run_experiment_grid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A utilizar dispositivo: {device}")
    
    dummy_model = BaselineCNN()
    total_params = sum(p.numel() for p in dummy_model.parameters())
    print(f"\n[INFO] Total parameters: {total_params} (Target: <1M)")

    # preparação dos dados
    trainloader, valloader, testloader, testset = setup_data()
    
    # Definição Configurações
    optimizers = ["SGD", "Adam"]
    learning_rates = [0.1, 0.01, 0.001]
    
    configs = []
    for opt in optimizers:
        for lr in learning_rates:
            configs.append({"opt": opt, "lr": lr})
            
    results = {}
    best_overall_val_acc = 0
    best_model_info = {} 
    best_model_preds = None
    best_config_name = ""
    
    num_epochs = 50 
    
    print(f"\n--- A iniciar Grid Search: {len(configs)} combinações ---")
    
    for config in configs:
        opt_name = config["opt"]
        lr = config["lr"]
        run_name = f"{opt_name} (lr={lr})"
        
        print(f"\n> A treinar: {run_name} ...")
        
        model = BaselineCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        
        if opt_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
        # Variáveis de histórico
        val_acc_history = []
        train_loss_history = []
        val_loss_history = []
        train_acc_history = [] 
        
        start_time = time.time()
        
        # Loop de Épocas
        for epoch in range(num_epochs):
            t_loss, t_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
            v_loss, v_acc, preds, labels = evaluate(model, valloader, criterion, device)
            
            val_acc_history.append(v_acc)
            val_loss_history.append(v_loss)
            train_loss_history.append(t_loss)
            train_acc_history.append(t_acc) 
        
        duration = time.time() - start_time
        
        final_val_acc = val_acc_history[-1]
        best_val_acc = max(val_acc_history)   
        best_train_acc = max(train_acc_history) 
        
        # Avaliação Final no Test Set
        _, test_acc, _, _ = evaluate(model, testloader, criterion, device)
        
        # Guardar resultados
        results[run_name] = {
            "val_acc": val_acc_history,           
            "train_loss": train_loss_history,     
            "time": duration,
            "best_train_acc": best_train_acc,     
            "final_train_loss": train_loss_history[-1],
            "best_val_acc": best_val_acc,         
            "test_acc": test_acc,
            "best_val_loss": min(val_loss_history),
            "final_val_loss": val_loss_history[-1]
        }
        
        print(f"  [{run_name}] Terminado | Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc
            best_model_preds = (preds, labels)
            best_config_name = run_name
            best_model_info = {
                "test_acc": test_acc,
                "best_train_acc": best_train_acc
            }

    epochs_range = range(1, num_epochs + 1)
    
    # Accuracy de Validação 
    plt.figure(figsize=(12, 6))
    for name, data in results.items():
        plt.plot(epochs_range, data["val_acc"], label=name, linewidth=2)
    plt.title('Comparação de Accuracy (Validação)')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    #Loss de Treino
    plt.figure(figsize=(12, 6))
    for name, data in results.items():
        plt.plot(epochs_range, data["train_loss"], label=name, linewidth=1.5)
    plt.title('Comparação de Loss de Treino (Diagnóstico de Instabilidade)')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Matriz de Confusão do melhor modelo
    print(f"\nMelhor Modelo (por Validação): {best_config_name}")
    print(f"  - Best Val Acc: {best_overall_val_acc:.4f}")
    print(f"  - Best Train Acc: {best_model_info['best_train_acc']:.4f}")
    print(f"  - Test Acc: {best_model_info['test_acc']:.4f}")

    if best_model_preds:
        best_preds, best_labels = best_model_preds
        cm = confusion_matrix(best_labels, best_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=testset.classes, yticklabels=testset.classes)
        plt.title(f'Matriz de Confusão - Melhor Modelo ({best_config_name})')
        plt.ylabel('Verdadeiro')
        plt.xlabel('Previsto')
        plt.show()
    
    header = f"{'Config':<20} | {'Time(m)':<8} | {'Best Train Acc':<15} | {'Train Final Loss':<18} | {'Best Val Acc':<15} | {'Test Acc':<10} | {'Val Best Loss':<15} | {'Val Final Loss':<15}"
    print(header)
    print("-" * len(header))
    
    for name, data in results.items():
        print(f"{name:<20} | "
              f"{data['time']/60:<8.2f} | "
              f"{data['best_train_acc']:<15.4f} | "
              f"{data['final_train_loss']:<18.4f} | "
              f"{data['best_val_acc']:<15.4f} | "
              f"{data['test_acc']:<10.4f} | "
              f"{data['best_val_loss']:<15.4f} | "
              f"{data['final_val_loss']:<15.4f}")
    print("="*140)

if __name__ == '__main__':
    try:
        run_experiment_grid()
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        input("Pressiona Enter para sair...")
