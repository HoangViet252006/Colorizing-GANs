import argparse
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from model import *
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description="Train GANs model")
    parser.add_argument("--root", "-d", type=str, default="./data", help="Path to Dataset")
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--checkpoint", "-s", type=str, default="trained_models/model.pt", help="Continue from this checkpoint")
    parser.add_argument("--num_workers", "-nw", type=int, default=3)

    args = parser.parse_args()
    return args

def test(args):
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(F"{device} was used")

    test_transform = ToTensor()


    test_dataset = torchvision.datasets.CIFAR10(root=args.root, train=False, download=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
    )

    generator = Generator().to(device)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        generator.load_state_dict(checkpoint["generator_params"])
    except Exception as e:
        print("Failed to load checkpoint:", e)
        exit(0)

    generator.eval()
    with (torch.no_grad()):
        for iter, (images, _) in enumerate(test_loader):
            # Convert the first 10 images to grayscale
            num_img = min(10, args.batch_size)
            sample_images = images[:num_img]
            sample_images_gray = rgb_to_gray(sample_images)
            sample_conditions = sample_images_gray.cpu()

            generations = generator(sample_conditions).cpu()
            generations = (generations + 1) / 2  # [0,1]
            generations = (generations * 255).clamp(0, 255).to(torch.uint8)
            generations = generations.squeeze(1)

            display_image_grid(sample_images, sample_images_gray, generations, num_img,
                               f"Batch {iter}", True, f"Batch_{iter}", "image")


if __name__ == '__main__':
    args = get_args()
    test(args)