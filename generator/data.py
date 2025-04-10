import torch
import torchvision
import torchvision.transforms as transforms

from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "data"

class TokenDataset(torch.utils.data.TensorDataset):
    def __init__(self, split: str):
        tensor_path = DATASET_PATH / f"{split}_tokenized.pth"
        if not tensor_path.exists():
            raise FileNotFoundError(
                f"Tokenized dataset not found at {tensor_path}. Create it following the assignment instructions."
            )
        self.data = torch.load(tensor_path, weights_only=False)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.data[idx], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

def load_data(in_path, img_sz=128, seed=123, train_split=0.95, batch_size=32, save_output=False, transform=False):
    # load images from kaggle data into torch dataset
    # apply resizing and cropping to images
    # split into training and testing set
    # return dataloaders
    if transform:
        transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128),transforms.ToTensor(),])
        data = torchvision.datasets.ImageFolder(in_path, transform=transform)
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        data = torchvision.datasets.ImageFolder(in_path, transform=transform)
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

def load_data_loader():
    train_loader = torch.load('./generator/data/trainloader.th', weights_only=False)
    val_loader = torch.load('./generator/data/trainloader.th', weights_only=False)
    return (train_loader, val_loader)