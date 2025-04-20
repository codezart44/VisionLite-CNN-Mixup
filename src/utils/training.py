import torch
from torch.utils.data import DataLoader
from typing import Callable
import gc


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


class EarlyStopper:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def check_stop(self, test_loss: float) -> bool:
        """ """
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience