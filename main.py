import argparse
import os.path
import torchvision
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, ColorJitter, ToTensor
from model import *
from epoch import *
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description="Train GANs model")
    parser.add_argument("--root", "-r", type=str, default="./data", help="Path to Dataset")
    parser.add_argument("--num_epochs", "-n", type=int, default=30)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--lr", "-l", type=float, default=2e-4, help="Learning rate for optimizer")
    parser.add_argument("--checkpoint_dir", "-c", type=str, default="trained_models", help="Path to save checkpoint")
    parser.add_argument("--checkpoint", "-s", type=str, default=None, help="Continue from this checkpoint")
    parser.add_argument("--l1_lambda", "-l1", type=int, default=100)
    parser.add_argument("--num_workers", "-nw", type=int, default=6)
    parser.add_argument("--is_download_data", "-i", type=bool, default=False)

    args = parser.parse_args()
    return args

def train(args):
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(F"{device} was used")

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)


    train_transform = Compose([
        ToTensor(),
        RandomRotation(5),
        RandomHorizontalFlip(0.1),
        ColorJitter(brightness=0.2),
    ])

    val_transform = ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(root=args.root, train=True, download=args.is_download_data, transform=train_transform)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
    )

    val_dataset = torchvision.datasets.CIFAR10(root=args.root, train=False, download=args.is_download_data, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_gen = Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_disc = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    scheduler = ExponentialLR(optimizer_gen, gamma=0.98)

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        generator.load_state_dict(checkpoint["generator_params"])
        discriminator.load_state_dict(checkpoint["discriminator_params"])
        optimizer_gen.load_state_dict(checkpoint["gen_optimizer"])
        optimizer_disc.load_state_dict(checkpoint["disc_optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        start_epoch = 1

    bce_criterion = nn.BCELoss()
    generator_losses_epoch_train, discriminator_losses_epoch_train = [], []
    generator_losses_epoch_val, discriminator_losses_epoch_val = [], []

    for epoch in range(start_epoch, args.num_epochs + 1):
        # Train
        generator_losses_epoch_train, \
        discriminator_losses_epoch_train = train_one_epoch(train_loader, args.batch_size, generator, discriminator, generator_losses_epoch_train,
                                                       discriminator_losses_epoch_train, args.l1_lambda, optimizer_gen, optimizer_disc,
                                                       bce_criterion, epoch, args.num_epochs, device)

        # val
        generator_losses_epoch_val, \
        discriminator_losses_epoch_val = validate_one_epoch(val_loader, args.batch_size, generator, discriminator, generator_losses_epoch_val,
                                                   discriminator_losses_epoch_val, args.l1_lambda, scheduler, bce_criterion, device)

        # save model
        if (epoch % 5 == 0):
            checkpoint = {
                "epoch": epoch + 1,
                "generator_params": generator.state_dict(),
                "discriminator_params": discriminator.state_dict(),
                "gen_optimizer": optimizer_gen.state_dict(),
                "disc_optimizer": optimizer_disc.state_dict(),
                "scheduler": scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"e{epoch}.pt"))

        # Visualization of validation images
        batch = next(iter(val_loader))
        images, labels = batch
        # Convert the first 10 images to grayscale
        num_img = min(10, args.batch_size)
        sample_images = images[:num_img]
        sample_images_gray = rgb_to_gray(sample_images)
        sample_conditions = sample_images_gray.to(device)

        generations = generator(sample_conditions).cpu()
        generations = (generations + 1) / 2  # [0,1]
        generations = (generations * 255).clamp(0, 255).to(torch.uint8)
        generations = generations.squeeze(1)

        display_image_grid(sample_images, sample_images_gray, generations, num_img,
                           f"First Batch Validate", False, None, None)

        scheduler.step()


if __name__ == '__main__':
    args = get_args()
    train(args)