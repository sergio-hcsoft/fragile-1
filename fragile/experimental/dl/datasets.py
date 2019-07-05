import torch
import torchvision as tv
import torchvision.transforms as transforms


class ImageDataset:
    def __init__(self, dataset=tv.datasets.MNIST):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )

        self.trainTransform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )

        self.trainset = dataset(root="./data", train=True, download=True, transform=self.transform)

        self.dataloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=32, shuffle=False, num_workers=4
        )

        self.testset = dataset(root="./data", train=False, download=True, transform=self.transform)

        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=4, shuffle=False, num_workers=2
        )
