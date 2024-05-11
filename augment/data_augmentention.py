from torch.utils.data import Dataset
import torchvision.transforms as transforms
# from cutout import Cutout
from augment.cutout import cutout
from augment.autoaugment_extra import CIFAR10Policy

# Reference https://github.com/hbzju/PiCO/blob/main/utils/cifar10.py
# Reference https://github.com/wu-dd/PLCR/blob/main/cifar.py
class CIFAR10_Augmentention(Dataset):
    def __init__(self, images, labelset, true_labels):
        self.images = images
        self.labelset = labelset
        self.true_labels = true_labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            cutout(n_holes=1, length=16),
            transforms.ToPILImage(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image_o = self.transform(self.images[index])
        each_image_w = self.transform1(self.images[index])
        each_image_s = self.transform2(self.images[index])
        each_label = self.labelset[index]
        each_true_label = self.true_labels[index]
        return each_image_o, each_image_w, each_image_s, each_label, each_true_label, index

# Reference https://github.com/hbzju/PiCO/blob/main/utils/cifar100.py
# Reference https://github.com/wu-dd/PLCR/blob/main/cifar.py
class CIFAR100_Augmentention(Dataset):
    def __init__(self, images, labelset, true_labels):
        self.images = images
        self.labelset = labelset
        self.true_labels = true_labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        self.transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
        ])
        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            cutout(n_holes=1, length=16),
            transforms.ToPILImage(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
        ])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image_o = self.transform(self.images[index])
        each_image_w = self.transform1(self.images[index])
        each_image_s = self.transform2(self.images[index])
        each_label = self.labelset[index]
        each_true_label = self.true_labels[index]
        return each_image_o, each_image_w, each_image_s, each_label, each_true_label, index