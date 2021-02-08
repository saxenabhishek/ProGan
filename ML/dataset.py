"""
Dataset

"""
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image


class Data():

    def __init__(self, path,  mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), size=(176,176), 
                xid=100, batch_size=128, shuffle=True, num_workers=2):
        self.path = path
        self.xid = xid
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


        self.transforms = transforms.Compose([
                          transforms.Resize(size),
                          transforms.ToTensor(), 
                          transforms.Normalize(mean, std)
        ])
        
        self.folderdata = ImageFolder(self.path, transform=self.transforms)
        


    def getdata(self):
        dataloader = DataLoader(self.folderdata, batch_size=self.batch_size, shuffle=self.shuffle, 
                                        num_workers=self.num_workers)
        return dataloader
        

    def showimage(self, xid):
        img = self.folderdata[xid]
        img = img[0]
        transforms.ToPILImage()(img).show()
        
        
if __name__ == "__main__":

    data = Data(path="../../fashiondata/img")
    alldata = data.getdata()