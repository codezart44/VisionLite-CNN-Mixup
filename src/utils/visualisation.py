import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def visualise_fashion_mnist(dataset: Dataset) -> None:
    """ """
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