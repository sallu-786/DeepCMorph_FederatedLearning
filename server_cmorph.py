# Copyright 2024 by Muhammad Suleman. All Rights Reserved.
import random
from typing import Dict, Optional, Tuple
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from sklearn.metrics import balanced_accuracy_score
from model import DeepCMorph  

# Constants
NUM_CLASSES = 9
BATCH_SIZE = 32
PATH_TO_TEST_DATASET = "data/CRC-VAL-HE-7K/"       #Only use this dataset for server
IMAGE_SIZE = 224

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main() :
    set_random_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize DeepCMorph model
    model = DeepCMorph(num_classes=NUM_CLASSES)  # Initialize the model
    model.load_weights(dataset="CRC")                                                   #load weights
   
   
   
    model.to(device)  #convert to cuda if available
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Convert model parameters to NDArrays for Flower
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_fn=get_evaluate_fn(model, device),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

def get_evaluate_fn(model, device):
    """Return an evaluation function for server-side evaluation."""
    
    test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  
        transforms.ToTensor(),
       # transforms.Normalize(mean=[0.729, 0.513, 0.715], std=[0.177, 0.236, 0.175])           #find mean and std deviation values for each dataset seperately do no copy paste
    ])
    
    val_dataset = datasets.ImageFolder(PATH_TO_TEST_DATASET, transform=test_transforms)
    val_loader = DataLoader(val_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False, 
                          num_workers=4, 
                          pin_memory=True)

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model parameters
        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
        
        model.to(device)
        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        targets_array = []
        predictions_array = []

        with tqdm(total=len(val_loader), desc="Evaluating", unit="batch") as pbar:
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    
                    predictions = predicted.detach().cpu().numpy()
                    targets_np = targets.detach().cpu().numpy()
                    
                    targets_array.extend(targets_np)
                    predictions_array.extend(predictions)
                    
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    pbar.update(1)

        accuracy = correct / total
        avg_loss = val_loss / len(val_loader)
        balanced_accuracy = balanced_accuracy_score(targets_array, predictions_array)

        write_to_csv(server_round, avg_loss, accuracy, balanced_accuracy)

        return avg_loss, {
            "classification accuracy": accuracy,
            "balanced accuracy": balanced_accuracy
        }

    return evaluate

def write_to_csv(server_round, loss, accuracy, balanced_accuracy):
    file_exists = os.path.isfile('evaluation_metrics_server.csv')
    with open('evaluation_metrics_server.csv', 'a') as f:
        if not file_exists:
            f.write('round,loss,accuracy,balanced_accuracy\n')
        f.write(f'{server_round},{loss},{accuracy},{balanced_accuracy}\n')

def fit_config(server_round: int):
    return {
        "batch_size": BATCH_SIZE,
        "local_epochs": 1
    }

def evaluate_config(server_round: int):
    return {"val_steps": 1}

if __name__ == "__main__":
    main() 
