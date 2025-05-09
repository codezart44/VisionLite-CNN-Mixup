import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
import gc
import time
from .globals import (
    EPOCHS,
    PATIENCE,
)
from .early_stopping import EarlyStopper
NUM_CLASSES = 10

def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """ Compute accuracy score of predicted labels. """
    y_pred = torch.argmax(y_pred, dim=1)
    accuracy = torch.sum(y_pred == y_true) / y_true.shape[0]
    return accuracy.item()

# def error_rate()  # NOTE - Just 1-accuracy, redundant

def confusion_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ Compute weighted confusion metrics recall and precision per class. """
    y_pred = torch.argmax(y_pred, dim=1)
    
    recall = torch.zeros((NUM_CLASSES, 2))
    precision = torch.zeros((NUM_CLASSES, 2))

    for cls in range(NUM_CLASSES):

        # Recall
        cls_arg = torch.argwhere(y_true == cls)
        arg_count = cls_arg.shape[0]
        if arg_count == 0:
            recall_score = torch.tensor([0])
        else:
            cls_pred = y_pred[cls_arg]
            cls_true = y_true[cls_arg]
            recall_score = torch.sum(cls_pred == cls_true) / cls_true.shape[0]
        recall[cls, :] = torch.tensor([recall_score*arg_count, arg_count])
        
        # Precision
        cls_arg = torch.argwhere(y_pred == cls)
        arg_count = cls_arg.shape[0]
        if arg_count == 0:
            precision_score = torch.tensor([0])
        else:
            cls_pred = y_pred[cls_arg]
            cls_true = y_true[cls_arg]
            precision_score = torch.sum(cls_true == cls_pred) / cls_pred.shape[0]
        precision[cls, :] = torch.tensor([precision_score*arg_count, arg_count])

    return recall, precision


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
    total_recall = torch.zeros((NUM_CLASSES, 2))
    total_precision = torch.zeros((NUM_CLASSES, 2))
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
        recall, precision = confusion_metrics(y, y_pred)
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy
        total_recall += recall
        total_precision += precision

        del x, y, y_pred, loss, recall, precision
    gc.collect()

    return (total_loss / num_batches,
            total_accuracy / num_batches,
            total_recall[:,0] / total_recall[:,1],        # normalize recalls
            total_precision[:,0] / total_precision[:,1])  # normalize precisions


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
    total_recall = torch.zeros((NUM_CLASSES, 2))
    total_precision = torch.zeros((NUM_CLASSES, 2))
    num_batches = len(dataloader)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x: torch.Tensor = batch[0]
            y: torch.Tensor = batch[1]
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            accuracy = accuracy_score(y, y_pred)
            recall, precision = confusion_metrics(y, y_pred)

            total_loss += loss.item()
            total_accuracy += accuracy
            total_recall += recall
            total_precision += precision

            del x, y, y_pred, loss, recall, precision
        gc.collect()

    return (total_loss / num_batches,
            total_accuracy / num_batches,
            total_recall[:,0] / total_recall[:,1],        # normalize recalls
            total_precision[:,0] / total_precision[:,1])  # normalize precisions


def train_eval_report(
        model : nn.Module,
        train_dataloader : DataLoader,
        test_dataloader : DataLoader,
        optimizer : torch.optim.Adam,
        loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device,
        early_stopper : EarlyStopper | None = None,
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
    if early_stopper is not None:
        early_stopper.reset_stopper()  # Reset inner count and best loss.
    model = model.to(device)

    train_losses = torch.zeros(EPOCHS)
    train_accuracies = torch.zeros(EPOCHS)
    train_times = torch.zeros(EPOCHS)
    train_recalls = torch.zeros((EPOCHS, NUM_CLASSES))
    train_precisions = torch.zeros((EPOCHS, NUM_CLASSES))

    test_losses = torch.zeros(EPOCHS)
    test_accuracies = torch.zeros(EPOCHS)
    test_times = torch.zeros(EPOCHS)
    test_recalls = torch.zeros((EPOCHS, NUM_CLASSES))
    test_precisions = torch.zeros((EPOCHS, NUM_CLASSES))

    # Train loop
    for epoch in range(EPOCHS):
        # Train one epoch
        start_train = time.time()
        # NOTE SANITY CHECK - Is mean train recall over classes about same as accuracy? 
        train_loss, train_acc, train_rcll, train_pres = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)
        end_train = time.time()
        # Train metrics
        train_losses[epoch] = train_loss
        train_accuracies[epoch] = train_acc
        train_times[epoch] = end_train - start_train
        train_recalls[epoch, :] = train_rcll
        train_precisions[epoch, :] = train_pres

        # Eval one epoch
        start_test = time.time()
        test_loss, test_acc, test_rcll, test_pres = evaluate(model, test_dataloader, loss_fn, device)
        end_test = time.time()
        # Test (inference) metrics
        test_losses[epoch] = test_loss
        test_accuracies[epoch] = test_acc
        test_times[epoch] = end_test - start_test
        test_recalls[epoch, :] = test_rcll
        test_precisions[epoch, :] = test_pres

        print(
            f"Epoch {epoch+1:3d} (Train/Test) " +\
            f"Loss: {train_loss:.6f}/{test_loss:.6f}. " +\
            f"Acc: {train_acc:.4f}/{test_acc:.4f}. " +\
            f"Rcll: {test_rcll}"
        )

        # print(
        #     f"Epoch {epoch+1:3d}  " +\
        #     f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}.  " +\
        #     f"Test Loss:  {test_loss:.6f},  Test Acc:  {test_acc:.4f}."
        # )

        if early_stopper is not None and early_stopper.check_stop(test_loss):
            print(f"Early Stopping at epoch {epoch+1}. Patience={PATIENCE}")
            break
    
    metrics_report = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'train_time': train_times,
        'train_recall': train_recalls, 
        'train_precision': train_precisions, 
        'test_loss': test_losses,
        'test_accuracy': test_accuracies,
        'test_time': test_times,
        'test_recall': test_recalls,
        'test_precisions': test_precisions,
    }

    return metrics_report
