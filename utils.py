import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def visualize_dataset(dataloader):
    batch = next(iter(dataloader))
    images = batch[0]
    labels = batch[1]
    plt.figure(figsize=(15, 15))
    for i in range(32):
        plt.subplot(8, 8, i + 1)
        img = np.transpose(images[i].numpy(), (1, 2, 0))  # Transpose the image dimensions
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis("off")
    plt.show()


def display_image_grid(original_images, gray_images, predicted_images, num_rows, title_text, is_save, image_name, save_path = "img"):
    """
    Visualize the images in a table with 3 columns:
        Column 1: Original images
        Column 2: Grayscale images
        Column 3: Predicted images
    Each row corresponds to the images in the three states mentioned above.
    """
    num_cols = 3
    fig = plt.figure(figsize=(num_cols, num_rows))
    grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_cols), axes_pad=0.15)
    for i, ax in enumerate(grid):
        # Determine which image to display based on the column
        idx = i // 3
        if i % 3 == 0:
            img = original_images[idx]
        elif i % 3 == 1:
            img = gray_images[idx]
        else:
            img = predicted_images[idx]

        # Display the image
        if img.size(0) == 1:   # grayscale
            if img.dtype == torch.float32 or img.dtype == torch.float64:
                ax.imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 1), cmap='gray')
            else:
                ax.imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 255), cmap='gray')
        else:  # RGB
            if img.dtype == torch.float32 or img.dtype == torch.float64:
                ax.imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 1))
            else:
                ax.imshow(np.clip(img.permute(1, 2, 0).numpy(), 0, 255))

        # Remove axis ticks
        ax.axis('off')

        # Set title for the first row of each column
        if i == 0:
            ax.set_title("Original", fontsize=10)
        elif i == 1:
            ax.set_title("Gray", fontsize=10)
        elif i == 2:
            ax.set_title("Predicted", fontsize=10)

    plt.suptitle(title_text, fontsize=16)
    plt.tight_layout()
    if is_save:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, image_name + ".png")
        plt.savefig(save_path, format='png', dpi=300)
    else:
        plt.show()

    plt.close('all')




def rgb_to_gray(batch):
    # Luminosity formula weights for R, G, B
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=batch.device)

    grayscale_batch = torch.tensordot(batch, weights, dims=([1], [0]))
    return grayscale_batch.unsqueeze(1)

def plot_losses(generator_losses_tr, discriminator_losses_tr, generator_losses_val, discriminator_losses_val):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Generator and Discriminator Losses', fontsize=16)

    # Plot Generator Training Loss
    axes[0, 0].plot(generator_losses_tr, label='Generator Training Loss', color='blue')
    axes[0, 0].plot(generator_losses_val, label='Generator Validation Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot Generator Validation Loss
    axes[0, 1].plot(discriminator_losses_tr, label='Discriminator Training Loss', color='blue')
    axes[0, 1].plot(discriminator_losses_val, label='Discriminator Validation Loss', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot Discriminator Training Loss
    axes[1, 0].plot(discriminator_losses_tr, label='Discriminator Training Loss', color='blue')
    axes[1, 0].plot(generator_losses_tr, label='Generator Training Loss', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot Discriminator Validation Loss
    axes[1, 1].plot(discriminator_losses_val, label='Discriminator Validation Loss', color='blue')
    axes[1, 1].plot(generator_losses_val, label='Generator Validation Loss', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Validation Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()
