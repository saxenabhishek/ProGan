"""
Dataset

"""
from typing import List, Tuple
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch


class Data:
    def __init__(
        self,
        path: str,
        mean: float = 0.5,
        std: float = 0.5,
        size1: Tuple[int, int] = (176, 176),
        size2: Tuple[int, int] = (64, 64),
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 2,
    ):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std),])

        # changed transforms to manips to avoid confusion
        self.folderdata = Twinimages(s1=size1, s2=size2, root_dir=self.path, manips=self.transforms)

    def changesize(self, s1: Tuple[int, int], s2: Tuple[int, int]):
        self.folderdata.ChangeSize(s1, s2)

    def getdata(self, split: List[int]):
        """
        split cannot contain 0

        If you want 2 splits only pass 0 in thrid index of split
        """

        def list_get(l, idx, default):
            try:
                return l[idx]
            except IndexError:
                return default

        split = [list_get(split, i, 0) for i in range(3)]

        divisor = sum(split)

        length = len(self.folderdata)
        print(length)
        print(split)

        # 1 is the samllest unit we can divide the dataset in.
        #  without the if condition it might be 0
        unit = length / divisor
        print(unit)
        # modifying split to be scaled according to legth of dataset
        split[0] = int(split[0] * unit)
        split[1] = int(split[1] * unit)
        if split[1] <= 0:
            split[1] = 1
            split[0] -= 1

        split[2] = length - split[0] - split[1]

        if split[2] <= 0:
            split[2] = 1
            split[0] -= 1

        print(split)

        assert sum(split) == length

        train, test, val = random_split(self.folderdata, split, generator=torch.Generator().manual_seed(69),)

        trainloader = DataLoader(
            train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )
        testloader = DataLoader(
            test, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )

        valloader = DataLoader(
            val, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )

        return trainloader, testloader, valloader

    def showimage(self, xid):
        img = self.folderdata[xid]
        img = img[0]
        transforms.ToPILImage()(img).show()


class Twinimages(Dataset):
    def __init__(self, s1, s2, root_dir, manips):
        self.path = root_dir
        self.manips = manips
        self.s1 = s1
        self.s2 = s2

        self.data = ImageFolder(self.path)

        self.Ts1 = transforms.Resize(self.s2)
        self.Ts2 = transforms.Resize(self.s2)

    def ChangeSize(self, s1, s2):
        self.Ts1 = transforms.Resize(s1)
        self.Ts2 = transforms.Resize(s2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        item_image = item[0]
        size1 = self.manips(self.Ts1(item_image))
        size2 = self.manips(self.Ts2(item_image))

        sample = {"S1": size1, "S2": size2, "class": [item[1]]}

        return sample


if __name__ == "__main__":
    data = Data(path="Data", batch_size=12)
    split = [19850, 50]
    train, test, val = data.getdata(split)
    data.changesize((64, 64), (12, 12))
    print(next(iter(test))["S1"].shape)
    print(len(train), len(test), len(val))
