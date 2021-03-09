"""
Dataset

"""
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch


class Data:
    def __init__(
        self, path, mean=0.5, std=0.5, size1=(176, 176), size2=(64, 64), batch_size=128, shuffle=True, num_workers=2,
    ):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std),])

        # changed transforms to manips to avoid confusion
        self.folderdata = Twinimages(s1=size1, s2=size2, root_dir=self.path, manips=self.transforms)

    def getdata(self, split):
        """
        split cannot contain 0

        If you want 2 splits only pass 0 in thrid index of split
        """
        assert len(split) == 3

        divisor = 0
        for i in range(3):
            divisor += split[i]

        length = len(self.folderdata)

        # 1 is the samllest unit we can divide the dataset in.
        #  without the if condition it might be 0
        unit = int(length / divisor) if divisor < length else 1

        # modifying split to be scaled according to legth of dataset
        split[0] = split[0] * unit
        split[1] = split[1] * unit

        split[2] = length - split[0] - split[1]

        train, test, val = random_split(self.folderdata, split, generator=torch.Generator().manual_seed(69),)

        trainloader = DataLoader(train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers,)
        testloader = DataLoader(test, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers,)

        valloader = DataLoader(val, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers,)

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        item_image = item[0]
        size1 = self.manips(transforms.Resize(self.s1)(item_image))
        size2 = self.manips(transforms.Resize(self.s2)(item_image))

        sample = {"S1": size1, "S2": size2, "class": [item[1]]}

        return sample


if __name__ == "__main__":
    data = Data(path="Data", batch_size=12)
    split = [5, 1, 0]
    train, test, val = data.getdata(split)
    print(test.dataset.dataset.s2)
    test.dataset.dataset.s2 = (12, 12)
    print(next(iter(test))["S2"].shape)
    # print(len(train), len(test), len(val))
