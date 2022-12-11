from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, STL10
import numpy as np

def get_transforms(name, args):
    train_transforms = {
        'cifar10': transforms.Compose([
                    transforms.RandomResizedCrop(224 if args.cc_data else 32),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
        'stl10' : transforms.Compose([
                    transforms.RandomResizedCrop(224 if args.cc_data else 96),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])

        
    }

    valid_transforms = {
        'cifar10': transforms.Compose([
                    transforms.Resize(224 if args.cc_data else 32),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
        'stl10': transforms.Compose([
                    transforms.Resize(224 if args.cc_data else 96),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])
    }
    return train_transforms[name], valid_transforms[name]




def get_contrastive_dataset(name, args):
    train_data = {
        'cifar10': CIFAR10Pair(root='data', train=True, transform=get_transforms('cifar10', args)[0], download=True),
        'stl10': STL10Pair(root='data', split='unlabeled', transform=get_transforms('stl10', args)[0], download=True)
    }
    
    test_data = {
        'cifar10': CIFAR10Pair(root='data', train=False, transform=get_transforms('cifar10', args)[1], download=True),
        'stl10': STL10Pair(root='data', split='test', transform=get_transforms('stl10', args)[1], download=True)

    }

    memory_data = {
        'cifar10': CIFAR10Pair(root='data', train=True, transform=get_transforms('cifar10', args)[1], download=True),
        'stl10': STL10Pair(root='data', split='unlabeled', transform=get_transforms('stl10', args)[1], download=True)

    }
    return train_data[name], memory_data[name], test_data[name]





class STL10Pair(STL10):
    """STL10 Dataset.
    """

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None        

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


