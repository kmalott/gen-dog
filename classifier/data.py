import torch
import torchvision
import torchvision.transforms as transforms

from pathlib import Path

def load_data(in_path, img_sz=128, seed=123, train_split=0.95, batch_size=32, save_output=False, transform=False, max_classes=120):
    # load images from kaggle data into torch dataset
    # apply resizing and cropping to images
    # split into training and testing set
    # return dataloaders
    if transform:
        transform = transforms.Compose([transforms.Resize(img_sz), transforms.CenterCrop(img_sz),transforms.ToTensor(),])
        data = torchvision.datasets.ImageFolder(in_path, transform=transform)
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        data = torchvision.datasets.ImageFolder(in_path, transform=transform)

    filtered_samples = [sample for sample in data.samples if sample[1] <= max_classes]
    data.samples = filtered_samples
    data.targets = [s[1] for s in filtered_samples]

    generator = torch.Generator().manual_seed(seed)
    split = torch.utils.data.random_split(data, [train_split, (1-train_split)], generator=generator)
    train_data = split[0]
    val_data = split[1]

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=batch_size)

    if save_output:
        torch.save(train_loader, './generator/data/trainloader.th')
        torch.save(val_loader, './generator/data/valloader.th')

    return (train_loader, val_loader)