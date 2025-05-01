import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from typing import Literal

def visualise_fashion_mnist(dataset: Dataset) -> None:
    """..."""
    samples = [dataset[i] for i in np.random.randint(0, len(dataset), 9)]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(6,6))
    fig.subplots_adjust(hspace=0, wspace=0.25)
    fig.suptitle('256x256 FashionMNIST Images')
    axes: np.ndarray
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax: plt.Axes
        image, label = samples[i]
        ax.imshow(np.squeeze(image), cmap='binary_r')
        ax.set_title(f'{dataset.classes[label]}') 
        ax.axis('off')
        
    plt.show()


def metric_ts(
        model_reports : dict[dict], 
        metric : Literal['loss', 'accuracy', 'time'],
        show_max_min : bool = True,
        sharey : bool = True,
        ) -> None:
    """..."""
    sns.set_style('whitegrid')
    palette = sns.color_palette('tab10')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,4), sharey=sharey)
    fig.suptitle(f'Comparing Model {metric.capitalize()}')
    ax1: plt.Axes  # Train axis
    ax2: plt.Axes  # Test axis

    for i, (name, report) in enumerate(model_reports.items()):
        color = palette[i]
        num_epochs = len(next(iter(report.values())))
        epochs = np.arange(1, num_epochs+1)
        train_metric = np.array(report['train_'+metric])
        test_metric = np.array(report['test_'+metric])

        ax1.plot(epochs, train_metric, c=color, label=name)
        ax2.plot(epochs, test_metric, c=color, label=name)

        if show_max_min == True:
            train_argmax = np.argmax(train_metric)
            test_argmax = np.argmax(test_metric)
            train_argmin = np.argmin(train_metric)
            test_argmin = np.argmin(test_metric)

            ax1.scatter(x=epochs[train_argmax], y=train_metric[train_argmax], c=[color], marker='X', s=80)
            ax1.scatter(x=epochs[train_argmin], y=train_metric[train_argmin], c=[color], marker='o', s=80)
            ax2.scatter(x=epochs[test_argmax], y=test_metric[test_argmax], c=[color], marker='X', s=80)
            ax2.scatter(x=epochs[test_argmin], y=test_metric[test_argmin], c=[color], marker='o', s=80)
        # plot a cross or some other small symbol at the place where the measure was max
        # same for min value

    ax1.set_title('Train')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(metric.capitalize())
    ax2.set_title('Test')
    ax2.set_xlabel('Epochs')
    ax2.legend()  # Same models are shown in both
    plt.show()
    sns.set_style('white')


def metric_bar(
        model_reports : dict[dict], 
        metric : Literal['loss', 'accuracy', 'time'],
        sharey : bool = True,
    ) -> None:
    """..."""
    sns.set_style('whitegrid')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,4), sharey=sharey)
    fig.suptitle(f'Comparing Model {metric.capitalize()}')
    ax1: plt.Axes  # Train axis
    ax2: plt.Axes  # Test axis
    model_names = list(model_reports.keys())
    num_models = len(model_reports)
    train_metric_max = np.zeros(num_models)
    test_metric_max = np.zeros(num_models)

    for i, (name, report) in enumerate(model_reports.items()):
        train_metric_max[i] = np.array(report['train_'+metric]).mean()
        test_metric_max[i] = np.array(report['test_'+metric]).mean()

    sns.barplot(ax=ax1, x=model_names, y=train_metric_max, palette='tab10')
    sns.barplot(ax=ax2, x=model_names, y=test_metric_max, palette='tab10')

    ax1.set_title('Train')
    ax1.set_xlabel('Model')
    ax1.set_ylabel(metric.capitalize())
    ax2.set_title('Test')
    ax2.set_xlabel('Model')
    ax2.legend()  # Same models are shown in both
    plt.show()
    sns.set_style('white')
