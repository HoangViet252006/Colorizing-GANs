import torch
from tqdm import tqdm
from model import *
from utils import rgb_to_gray

def train_one_epoch(train_loader, batch_size, generator, discriminator, gen_losses_train, disc_losses_train,
                    l1_lambda, optimizer_gen, optimizer_disc, bce_criterion, current_epoch, total_epochs, device):
    discriminator.train()
    generator.train()

    l1_loss = nn.L1Loss()

    total_generator_loss = 0.0
    total_discriminator_loss = 0.0
    num_samples = batch_size * len(train_loader)

    pbar = tqdm(train_loader, colour="cyan")
    for iteration, (images, _) in enumerate(pbar):
        images = images.to(device)
        grayscale_images = rgb_to_gray(images)
        condition = grayscale_images.to(device)

        # Normalize to [-1, 1]
        condition = (condition - 0.5) / 0.5
        images = (images - 0.5) / 0.5

        # Discriminator

        discriminator.zero_grad()

        # real, fake label shape equal discriminator shape

        output_real = discriminator(images, condition)
        real_label = torch.ones_like(output_real).to(device)
        loss_real = bce_criterion(output_real, real_label)

        fake_images = generator(condition)

        output_fake = discriminator(fake_images.detach(), condition)
        fake_label = torch.zeros_like(output_fake).to(device)
        loss_fake = bce_criterion(output_fake, fake_label)

        loss_discriminator = loss_real + loss_fake
        loss_discriminator.backward()
        optimizer_disc.step()

        # Generator
        generator.zero_grad()
        output_fake = discriminator(fake_images, condition)
        loss_generator = bce_criterion(output_fake, real_label) + (l1_lambda * l1_loss(fake_images, images))
        loss_generator.backward()
        optimizer_gen.step()

        # Update metrics
        total_generator_loss += loss_generator.item()
        total_discriminator_loss += loss_discriminator.item()

        pbar.set_description(f"Epoch {current_epoch}/{total_epochs}. Generator_loss: {loss_generator.item():.4f}. "
                             f"Discriminator_loss: {loss_discriminator.item():.4f}")

    gen_losses_train.append(total_generator_loss / num_samples)
    disc_losses_train.append(total_discriminator_loss / num_samples)

    return gen_losses_train, disc_losses_train


def validate_one_epoch(val_loader, batch_size, generator, discriminator, gen_losses_val, disc_losses_val,
                       l1_lambda, scheduler, bce_criterion, device):
    generator.eval()
    discriminator.eval()
    l1_loss = nn.L1Loss()
    total_generator_loss = 0.0
    total_discriminator_loss = 0.0
    num_samples = batch_size * len(val_loader)
    pbar = tqdm(val_loader, colour="green")

    with torch.no_grad():
        for iteration, (images, _) in enumerate(pbar):

            grayscale_images = rgb_to_gray(images)
            condition = grayscale_images.to(device)
            images = images.to(device)

            # Normalize to [-1, 1]
            condition = (condition - 0.5) / 0.5
            images = (images - 0.5) / 0.5

            # Discriminator
            # real, fake label shape equal discriminator shape

            output_real = discriminator(images, condition)
            real_label = torch.ones_like(output_real).to(device)
            loss_real = bce_criterion(output_real, real_label)

            fake_images = generator(condition)

            output_fake = discriminator(fake_images.detach(), condition)
            fake_label = torch.zeros_like(output_fake).to(device)
            loss_fake = bce_criterion(output_fake, fake_label)

            loss_discriminator = loss_real + loss_fake

            # Generator
            output_fake = discriminator(fake_images, condition)
            loss_generator = bce_criterion(output_fake, real_label) + (l1_lambda * l1_loss(fake_images, images))

            # Update metrics
            total_generator_loss += loss_generator.item()
            total_discriminator_loss += loss_discriminator.item()

            pbar.set_description(f"Generator_loss: {loss_generator.item():.4f}. Discriminator_loss: {loss_discriminator.item():.4f}")

        gen_losses_val.append(total_generator_loss / num_samples)
        disc_losses_val.append(total_discriminator_loss / num_samples)

    scheduler.step()
    return gen_losses_val, disc_losses_val
