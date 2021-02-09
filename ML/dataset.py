"""
Dataset

"""
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch


class Data:
    def __init__(
        self,
        path,
        trainset,
        testset,
        valset,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        size=(176, 176),
        xid=100,
        batch_size=128,
        shuffle=True,
        num_workers=2,
    ):
        self.path = path
        self.xid = xid
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.trainset = trainset

        self.valset = valset
        self.testset = testset

        self.transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.folderdata = ImageFolder(self.path, transform=self.transforms)

    def getdata(self):

        train, test, val = random_split(
            self.folderdata,
            [self.trainset, self.testset, self.valset],
            generator=torch.Generator().manual_seed(42),
        )

        trainloader = DataLoader(
            train, shuffle=True, batch_size=self.batch_size, num_workers=2
        )
        testloader = DataLoader(
            test, shuffle=True, batch_size=self.batch_size, num_workers=2
        )

        valloader = DataLoader(
            val, shuffle=True, batch_size=self.batch_size, num_workers=2
        )

        return trainloader, testloader, valloader

    def showimage(self, xid):
        img = self.folderdata[xid]
        img = img[0]
        transforms.ToPILImage()(img).show()


if __name__ == "__main__":
    data = Data(
        path="../../fashiondata/img", trainset=50000, testset=2000, valset=712
    )
    train, test, val = data.getdata()

