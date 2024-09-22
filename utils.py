from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

def get_dataset(batch_size = 64):
    transform = Compose([
        Resize((448, 448)),
        ToTensor()
    ])

    train_dataset = datasets.VOCDetection(root = 'data', year = '2007', image_set = 'train', download = True, transform = transform)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    return train_loader

from torchvision import transforms

class Resize(object):
    """
    Resize the input PIL.Image to the given size.
    """
    def init(self, size=(448, 448)):
        self.size = size

    def call(self, image, boxes):
        return image.resize(self.size), boxes

class ToTensor(object):
    """
    Convert PIL.Image to Tensor.
    """
    def call(self, image, boxes):
        image = transforms.ToTensor()(image)
        return image, boxes

class Compose(object):
    """
    Compose multiple transforms together.
    """
    def init(self, transforms):
        self.transforms = transforms

    def call(self, image, boxes):
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes