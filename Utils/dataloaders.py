import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image

from typing import Optional, Callable, Type
import platform
import pandas as pd
from PIL import Image  #


def prepare_fashion_mnist(train_compose: transforms.Compose,
                          test_compose: transforms.Compose,
                          save_path: str,
                          batch_size: int = 64,
                          num_workers: int = 4):
    """
    Prepare Fashion MNIST dataset with custom transformations. To avoid multi
    :param train_compose: Compose object for training data
    :param test_compose: Compose object for test data
    :param save_path: Path to save dataset
    :param batch_size: Batch size
    :param num_workers: Number of workers
    :return: train_loader, test_loader, class_names

    """
    # Set appropriate num_workers based on OS
    if num_workers:
        if platform.system() == 'Windows':
            # Use 0 workers on Windows by default to avoid multiprocessing issues:
            # Windows uses "spawn" for creating processes instead of "fork", which can cause issues with multiprocessing
            num_workers = 0

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Load training data with custom transformations
    train_dataset = datasets.FashionMNIST(
        root=save_path,
        train=True,
        download=True,
        transform=train_compose
    )

    # Load test data with basic transformations
    test_dataset = datasets.FashionMNIST(
        root=save_path,
        train=False,
        download=True,
        transform=test_compose
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Fashion MNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return train_loader, test_loader, class_names


def make_annotation_file(path_train: str, path_test: str, path_val: str, labels: list[str]):
    ## function to iterate through the path and create an annotation csv
    list_train = []
    list_test = []
    list_val = []
    for idx, label in enumerate(labels):
        for file in os.listdir(path_train + label):
            list_train.append((label + '/' + file, idx))

        for file in os.listdir(path_test + label):
            list_test.append((label + '/' + file, idx))

        for file in os.listdir(path_val + label):
            list_val.append((label + '/' + file, idx))

    ## create two csv-files for the annotations
    csv_train = pd.DataFrame(list_train, columns=['', ''])
    csv_test = pd.DataFrame(list_test, columns=['', ''])
    csv_val = pd.DataFrame(list_val, columns=['', ''])

    csv_train.to_csv(path_train + 'annot_train.csv', sep=',', index=False)
    csv_test.to_csv(path_test + 'annot_test.csv', sep=',', index=False)
    csv_val.to_csv(path_val + 'annot_val.csv', sep=',', index=False)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def prepare_UTKFace_age_task(
    train_compose: transforms.Compose,
    test_compose: transforms.Compose,
    data_path: str = '../Dataset/UTKFace_aligned_paper_split_classification_4classes',
    batch_size: int = 64,
    num_workers: int = 4
):
    # Set appropriate num_workers based on OS
    if num_workers:
        if platform.system() == 'Windows':
            num_workers = 0

    train_data_path = f'{data_path}/train/classes/'
    test_data_path = f'{data_path}/test/classes/'
    val_data_path = f'{data_path}/val/classes/'
    classes = ['0-17', '18-40', '41-60', '61-200']

    try:
        make_annotation_file(train_data_path, test_data_path, val_data_path, classes)

        training_data = CustomImageDataset(train_data_path + 'annot_train.csv', train_data_path, transform=train_compose)
        trainloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        test_data = CustomImageDataset(test_data_path + 'annot_test.csv', test_data_path, transform=test_compose)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        return trainloader, testloader, classes

    except FileNotFoundError as e:
        print('Annotation files not found. Please make sure you downloaded the dataset and unzipped it in the given path.')
        raise e


class TinyImageNetDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):

        """
        :param root_dir: Directory with Tiny ImageNet data
        :param split: 'train', 'val', or 'test'
        :param transform: Transform to be applied on images. Default: No transformations
        """

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Load class IDs and class names
        self.load_class_info()

        # Load image paths and labels for the specified split
        self.images, self.labels = self.load_split_data()

    def load_class_info(self):
        """Load class information from wnids.txt and words.txt"""
        # Read class IDs (wnids)
        wnids_path = os.path.join(self.root_dir, 'wnids.txt')
        if os.path.exists(wnids_path):
            wnids_df = pd.read_csv(wnids_path, header=None, names=['wnid'])
            self.classes = wnids_df['wnid'].tolist()
            self.class_to_idx = {wnid: i for i, wnid in enumerate(self.classes)}
            self.idx_to_class = {i: wnid for i, wnid in enumerate(self.classes)}
        else:
            self.classes = []
            self.class_to_idx = {}
            self.idx_to_class = {}

        # Read human-readable class names
        words_path = os.path.join(self.root_dir, 'words.txt')
        if os.path.exists(words_path):
            words_df = pd.read_csv(words_path, sep='\t', header=None, names=['wnid', 'name'])
            self.class_names = dict(zip(words_df['wnid'], words_df['name']))
        else:
            self.class_names = {}

    def load_split_data(self):
        """Load image paths and labels for the specified split"""
        images = []
        labels = []

        if self.split == 'train':
            # Process training data
            train_dir = os.path.join(self.root_dir, 'train')
            for wnid in os.listdir(train_dir):
                if wnid in self.class_to_idx:
                    class_idx = self.class_to_idx[wnid]
                    img_dir = os.path.join(train_dir, wnid, 'images')
                    if os.path.isdir(img_dir):
                        for img_file in os.listdir(img_dir):
                            if img_file.endswith('.JPEG'):
                                images.append(os.path.join(img_dir, img_file))
                                labels.append(class_idx)

        elif self.split == 'val':
            # Process validation data
            val_dir = os.path.join(self.root_dir, 'val')
            val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

            if os.path.exists(val_annotations_file):
                # Read validation annotations as DataFrame, we just need the first 2 columns the rest is for bounding boxes
                val_df = pd.read_csv(val_annotations_file, sep='\t', header=None,
                                     names=['filename', 'wnid', 'x', 'y', 'h', 'w'])

                # Filter by valid class IDs and existing files
                for _, row in val_df.iterrows():
                    if row['wnid'] in self.class_to_idx:
                        img_path = os.path.join(val_dir, 'images', row['filename'])
                        if os.path.exists(img_path):
                            images.append(img_path)
                            labels.append(self.class_to_idx[row['wnid']])

        elif self.split == 'test':
            # Process test data (no labels)
            test_dir = os.path.join(self.root_dir, 'test')
            test_img_dir = os.path.join(test_dir, 'images')

            if os.path.isdir(test_img_dir):
                for img_file in os.listdir(test_img_dir):
                    if img_file.endswith('.JPEG'):
                        images.append(os.path.join(test_img_dir, img_file))
                        labels.append(-1)  # No labels for test data

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Load image
        image = read_image(img_path)
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_class_name(self, idx):
        """Get human-readable class name for a given class index"""
        if idx == -1:
            return "unknown"

        wnid = self.idx_to_class.get(idx)
        if wnid and wnid in self.class_names:
            return self.class_names[wnid]
        return wnid or f"class_{idx}"


def prepare_tiny_imagenet(
        train_transform: transforms.Compose,
        test_transform: transforms.Compose,
        data_path: str,
        batch_size: int = 256,
        num_workers: int = 4):
    """
    Create dataloaders for Tiny ImageNet
    """
    # Create datasets
    train_dataset = TinyImageNetDataset(
        root_dir=data_path,
        split='train',
        transform=train_transform
    )

    val_dataset = TinyImageNetDataset(
        root_dir=data_path,
        split='val',
        transform=test_transform
    )

    test_dataset = TinyImageNetDataset(
        root_dir=data_path,
        split='test',
        transform=test_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Return dataloaders along with class information
    return train_loader, val_loader, test_loader, {
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx,
        'idx_to_class': train_dataset.idx_to_class,
        'class_names': train_dataset.class_names,
    }
