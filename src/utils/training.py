import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
import gc
import time
from .globals import (
    EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    BATCH_SIZE,
)

class EarlyStopper:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def check_stop(self, test_loss: float) -> bool:
        """..."""
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
    
    def reset_stopper(self) -> None:
        """..."""
        self.best_loss = float('-inf')
        self.counter = 0

# Reusables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
early_stopper = EarlyStopper(patience=PATIENCE)

def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """ Compute accuracy score of predicted labels. """
    y_pred = torch.argmax(y_pred, dim=1)
    accuracy = torch.sum(y_pred == y_true) / y_true.shape[0]
    return accuracy.item()

def train_one_epoch(
        model: torch.nn.Module, 
        dataloader: DataLoader,
        optimizer: torch.optim.Adam, 
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        device
    ) -> tuple[float, float]:
    """ """
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = len(dataloader)

    for i, batch in enumerate(dataloader):
        x: torch.Tensor = batch[0]
        y: torch.Tensor = batch[1]
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred: torch.Tensor = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        accuracy = accuracy_score(y, y_pred)
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy

        del x, y, y_pred, loss
    gc.collect()

    return total_loss / num_batches, total_accuracy / num_batches


def evaluate(
        model: torch.nn.Module,
        dataloader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device
    ) -> tuple[float, float]:
    """ """
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x: torch.Tensor = batch[0]
            y: torch.Tensor = batch[1]
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            accuracy = accuracy_score(y, y_pred)

            total_loss += loss.item()
            total_accuracy += accuracy

            del x, y, y_pred, loss
        gc.collect()

    return total_loss / num_batches, total_accuracy / num_batches


def train_eval_report(
        model : nn.Module,
        train_dataloader : DataLoader,
        test_dataloader : DataLoader,
        optimizer : torch.optim.Adam,
        loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> dict:
    """
    Training and inference loop to collect model metrics.

    Parameters
    ----------
    model : Module
        ...
    train_dataloader : DataLoader
        ...
    test_dataloader : DataLoader
        ...
    optimizer : Adam
        ...
    loss_fn : Callable[[Tensor, Tensor], Tensor]

    Returns
    -------
    dict
        Report of train and test/inference: accuracy, loss and time
    """
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=PATIENCE)

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    train_times, inference_times = [], []

    # Train loop
    for epoch in range(EPOCHS):
        # Train one epoch
        start_train = time.time()
        train_loss, train_acc = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)
        end_train = time.time()
        # Train metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_times.append(end_train - start_train)

        # Eval one epoch
        start_infer = time.time()
        test_loss, test_acc = evaluate(model, train_dataloader, loss_fn, device)
        end_infer = time.time()
        # Test and inference metrics
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        inference_times.append(end_infer - start_infer)

        print(
            f"Epoch {epoch+1:3d}  " +\
            f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}.  " +\
            f"Test Loss:  {test_loss:.6f},  Test Acc:  {test_acc:.4f}."
        )

        if early_stopper.check_stop(test_loss):
            print(f"Early Stopping at epoch {epoch+1}. Patience={PATIENCE}")
            break
    
    metrics_report = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'train_times': train_times,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'inference_times': inference_times
    }

    return metrics_report
