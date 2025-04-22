import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from data_loader import load_cifar10_data

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging
log_file = os.path.join('logs', f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')


def log(message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')
    print(message)


# Function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Step 2: Define the CNN architecture with Dropout
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Step 3: Compile and train the model with learning rate scheduler
def train_model(model, trainloader, testloader, epochs=30, lr=0.0001,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # Reduce LR by factor of 0.1 every 10 epochs

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(testloader)
        val_acc = 100 * correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        log(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}')

    return history


# Step 4: Plot the loss curve with hyperparameters
def plot_loss_curve(history, hyperparameters):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Add hyperparameters as text on the plot
    hyperparam_text = '\n'.join([f'{key}: {value}' for key, value in hyperparameters.items()])
    plt.text(0.02, 0.98, hyperparam_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig('loss_curve.png')
    log('Loss curve saved as loss_curve.png')


# Step 5: Evaluate the model
def evaluate_model(model, testloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    log(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


# Step 6: Analyze hyperparameters
def analyze_hyperparameters(trainloader, testloader, hyperparameters,
                            device='cuda' if torch.cuda.is_available() else 'cpu'):
    log('\n=== Hyperparameter Analysis ===')

    class CNNModifiedFilters(nn.Module):
        def __init__(self):
            super(CNNModifiedFilters, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 4 * 4, 256)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
            x = x.view(-1, 64 * 4 * 4)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    log('Training with reduced filter sizes (16, 32, 64)...')
    model_filters = CNNModifiedFilters().to(device)
    history_filters = train_model(model_filters, trainloader, testloader, epochs=30, lr=hyperparameters['learning_rate'])
    acc_filters = evaluate_model(model_filters, testloader)
    log(f'Accuracy with reduced filters: {acc_filters:.2f}%')

    log('Training with higher learning rate (0.01)...')
    model_lr = CNN().to(device)
    history_lr = train_model(model_lr, trainloader, testloader, epochs=30, lr=0.01)
    acc_lr = evaluate_model(model_lr, testloader)
    log(f'Accuracy with learning rate 0.01: {acc_lr:.2f}%')

    class CNNNoBatchNorm(nn.Module):
        def __init__(self):
            super(CNNNoBatchNorm, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(128 * 4 * 4, 256)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = self.pool1(torch.relu(self.conv1(x)))
            x = self.pool2(torch.relu(self.conv2(x)))
            x = self.pool3(torch.relu(self.conv3(x)))
            x = x.view(-1, 128 * 4 * 4)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    log('Training without BatchNorm...')
    model_no_bn = CNNNoBatchNorm().to(device)
    history_no_bn = train_model(model_no_bn, trainloader, testloader, epochs=30, lr=hyperparameters['learning_rate'])
    acc_no_bn = evaluate_model(model_no_bn, testloader)
    log(f'Accuracy without BatchNorm: {acc_no_bn:.2f}%')


# Step 7: Transfer Learning with ResNet-50
def transfer_learning(trainloader, testloader, lr=0.0001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    log('\n=== Transfer Learning with ResNet-50 ===')

    resnet = models.resnet50(pretrained=True)
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = nn.Identity()
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)
    resnet = resnet.to(device)
    history_resnet = train_model(resnet, trainloader, testloader, epochs=30, lr=lr)
    acc_resnet = evaluate_model(resnet, testloader)
    log(f'Accuracy with ResNet-50 (Transfer Learning): {acc_resnet:.2f}%')


# Main execution
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define hyperparameters
    hyperparameters = {
        'batch_size': 32,
        'epochs': 30,
        'learning_rate': 0.0001,
        'weight_decay': 1e-4,
        'dropout_rate': 0.5,
        'scheduler_step_size': 10,
        'scheduler_gamma': 0.1
    }

    # Log initial setup
    log(f'Using device: {device}')
    log('\n=== Hyperparameters ===')
    for key, value in hyperparameters.items():
        log(f'{key}: {value}')

    # Step 1: Load data
    trainloader, testloader = load_cifar10_data(batch_size=hyperparameters['batch_size'])

    # Step 2 & 3: Define and compile the model
    model = CNN()
    num_params = count_parameters(model)
    log(f'\n=== Model Parameters ===')
    log(f'Number of trainable parameters: {num_params}')

    log('\n=== Training Custom CNN ===')
    history = train_model(model, trainloader, testloader, epochs=hyperparameters['epochs'],
                          lr=hyperparameters['learning_rate'])

    # Step 4: Plot loss curve with hyperparameters
    plot_loss_curve(history, hyperparameters)

    # Step 5: Evaluate the model
    log('\n=== Model Evaluation ===')
    test_accuracy = evaluate_model(model, testloader)

    # Step 6: Analyze hyperparameters
    analyze_hyperparameters(trainloader, testloader, hyperparameters)

    # Step 7: Transfer Learning with ResNet-50
    transfer_learning(trainloader, testloader, lr=hyperparameters['learning_rate'])