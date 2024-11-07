# Copyright 2024 by Muhammad Suleman. All Rights Reserved.
import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import csv
from collections import defaultdict
from model import DeepCMorph  


NUM_CLASSES = 9
BATCH_SIZE = 32
IMAGE_SIZE = 224
PATH_TO_TRAIN_DATASET = "data/CRC-VAL-HE-7K/"  # INST1: NCT-CRC-HE-100K-->download this dataset and use 30,000 unique images for each client instead of 7K which is for testing
PATH_TO_TEST_DATASET = "data/CRC-VAL-HE-7K/"   #NCT-CRC-HE-100K-->downloadthis dataset and use 3300 of its uniques images as test dataset for each client. make sue all classes are present

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DeepCMorphClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.metrics = defaultdict(list)

    def get_parameters(self):
        """Get parameters of the local model."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Set local model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        self.set_parameters(parameters)
        
        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Define optimizer and loss function
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        self.model.train()
        total_loss = 0.0
        for epoch in range(epochs):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

        # Calculate average loss
        avg_loss = total_loss / (len(self.train_loader) * epochs)
        self.metrics['train_loss'].append(avg_loss)

        return self.get_parameters(), len(self.train_loader.dataset), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()

        # Evaluate the model
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total
        avg_loss = loss / len(self.test_loader)
        
        self.metrics['test_loss'].append(avg_loss)
        self.metrics['test_accuracy'].append(accuracy)

        return loss, total, {"accuracy": accuracy}

    def save_metrics_to_csv(self, filename='metrics_client_aus.csv'):
        """Save recorded metrics to a CSV file."""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Round', 'Train Loss', 'Test Loss', 'Test Accuracy'])
            for i in range(len(self.metrics['train_loss'])):
                writer.writerow([
                    i + 1,
                    self.metrics['train_loss'][i],
                    self.metrics['test_loss'][i],
                    self.metrics['test_accuracy'][i]
                ])
        print(f"Metrics saved to {filename}")

def load_data():
    """Load and prepare the dataset."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # INST2: Calculate mean and std for this client dataset and use those values here
    ])

    # Load training and validation datasets
    train_dataset = datasets.ImageFolder(
        PATH_TO_TRAIN_DATASET,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
   
    test_dataset = datasets.ImageFolder(
        PATH_TO_TEST_DATASET,  
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader

def main() -> None:
    # Set random seed
    set_random_seed(42)
    
    # Initialize model
    model = DeepCMorph(num_classes=NUM_CLASSES)
    model.load_weights(dataset="CRC")                    #INST3: First try without loading any weights, then uncomment this line for loading weights of relevant dataset    

    train_loader, test_loader = load_data()

    # Start Flower client
    client = DeepCMorphClient(model, train_loader, test_loader)
    

    fl.client.start_client(
        server_address="localhost:8080",
        client=client,
    )
    
    # Save metrics after training
    client.save_metrics_to_csv()

if __name__ == "__main__":
    main()
