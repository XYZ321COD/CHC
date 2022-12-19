from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, STL10, ImageNet, MNIST, FashionMNIST
from torch.utils.data import Subset
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
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])]),

        'imagenet10': transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        'imagenetdogs': transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        'mnist': transforms.Compose([
                        transforms.RandomResizedCrop(28),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))]),
        'fmnist': transforms.Compose([
                        transforms.RandomResizedCrop(28),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])
   
    }

    valid_transforms = {
        'cifar10': transforms.Compose([
                    transforms.Resize(224 if args.cc_data else 32),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
        'stl10': transforms.Compose([
                    transforms.Resize(224 if args.cc_data else 96),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])]),
        'imagenet10': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'imagenetdogs': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'mnist': transforms.Compose([
                        transforms.Resize(28),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))]),
        'fmnist': transforms.Compose([
                        transforms.Resize(28),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])
    }
    return train_transforms[name], valid_transforms[name]




def get_contrastive_dataset(name, args):
    train_data = {
        'cifar10': CIFAR10Pair(root='data', train=True, transform=get_transforms('cifar10', args)[0], download=True),
        'stl10': STL10Pair(root='data', split='unlabeled', transform=get_transforms('stl10', args)[0], download=True),
        # 'imagenet10': filter_ImageNet(ImageNetPair('/shared/sets/datasets/vision/ImageNet', split='train', transform=get_transforms('imagenet10', args)[0]), args), 
        # 'imagenetdogs': filter_ImageNet(ImageNetPair('/shared/sets/datasets/vision/ImageNet', split='train', transform=get_transforms('imagenetdogs', args)[0]), args), 
        'mnist': MNISTPair(root='data', train=True, transform=get_transforms('mnist', args)[0], download=True),
        'fmnist': FashionMNISTPair(root='data', train=True, transform=get_transforms('fmnist', args)[0], download=True)

    }
    
    test_data = {
        'cifar10': CIFAR10Pair(root='data', train=False, transform=get_transforms('cifar10', args)[1], download=True),
        'stl10': STL10Pair(root='data', split='test', transform=get_transforms('stl10', args)[1], download=True),
        # 'imagenet10': filter_ImageNet(ImageNetPair('/shared/sets/datasets/vision/ImageNet', split='val', transform=get_transforms('imagenet10', args)[1]), args),
        # 'imagenetdogs': filter_ImageNet(ImageNetPair('/shared/sets/datasets/vision/ImageNet', split='val', transform=get_transforms('imagenetdogs', args)[1]), args),
        'mnist': MNISTPair(root='data', train=False, transform=get_transforms('mnist', args)[1], download=True),
        'fmnist': FashionMNISTPair(root='data', train=False, transform=get_transforms('fmnist', args)[1], download=True)




    }

    memory_data = {
        'cifar10': CIFAR10Pair(root='data', train=True, transform=get_transforms('cifar10', args)[1], download=True),
        'stl10': STL10Pair(root='data', split='train', transform=get_transforms('stl10', args)[1], download=True),
        # 'imagenet10': filter_ImageNet(ImageNetPair('/shared/sets/datasets/vision/ImageNet', split='train', transform=get_transforms('imagenet10', args)[1]), args),
        # 'imagenetdogs': filter_ImageNet(ImageNetPair('/shared/sets/datasets/vision/ImageNet', split='train', transform=get_transforms('imagenetdogs', args)[1]), args),
        'mnist': MNISTPair(root='data', train=True, transform=get_transforms('mnist', args)[1], download=True),
        'fmnist': FashionMNISTPair(root='data', train=True, transform=get_transforms('fmnist', args)[1], download=True)


    }
    return train_data[name], memory_data[name], test_data[name]


def filter_ImageNet(image_dataset, args):
    if args.dataset_name == 'imagenet10':
            subset_winds = [
                "n02056570",
                "n02085936",
                "n02128757",
                "n02690373",
                "n02692877",
                "n03095699",
                "n04254680",
                "n04285008",
                "n04467665",
                "n07747607"
            ]
    elif args.dataset_name == 'imagenetdogs':
            subset_winds = [
                "n02085936",
                "n02086646",
                "n02088238",
                "n02091467",
                "n02097130",
                "n02099601",
                "n02101388",
                "n02101556",
                "n02102177",
                "n02105056",
                "n02105412",
                "n02105855",
                "n02107142",
                "n02110958",
                "n02112137"
            ]
    elif args.dataset_name == 'imagenet30':
            subset_winds = [
            "n12267677",
            "n02690373",
            "n02701002",
            "n01698640",
            "n02787622",
            "n02793495",
            "n02837789",
            "n03196217",
            "n02268443",
            "n03255030",
            "n03384352",
            "n03443371",
            "n03452741",
            "n07697537",
            "n03544143",
            "n03717622",
            "n03788195",
            "n03804744",
            "n03891332",
            "n03938244",
            "n04086273",
            "n03187595",
            "n04147183",
            "n04252077",
            "n04254680",
            "n01498041",
            "n07745940",
            "n04389033",
            "n04442312",
            "n09472597"]
    subset_idx = [idx for idx, target in enumerate(image_dataset.wnids) if target in subset_winds]
    subset_indices = [idx for idx, target in enumerate(image_dataset.targets) if target in subset_idx]
    class_names = image_dataset.classes
    classes = tuple([class_names[c][0] for c in subset_idx])
    image_dataset = Subset(image_dataset, subset_indices)
    image_dataset.subset_index_attr = subset_idx
    return image_dataset

class ImageNetPair(ImageNet):
    """ImageNet Dataset.
    """

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return pos_1, pos_2, target



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



class MNISTPair(MNIST):
    """MNIST Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class FashionMNISTPair(FashionMNIST):
    """FashionMNIST Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target
