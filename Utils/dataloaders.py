import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.datasets.voc import VOCDetection
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

import platform
import pandas as pd


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


def prepare_imagenette(train_compose: transforms.Compose,
                          test_compose: transforms.Compose,
                          save_path: str,
                          batch_size: int = 64,
                          num_workers: int = 4):
    """
    Prepare the Imagenette  dataset with custom transformations.
    Find more information here: https://github.com/fastai/imagenette

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

    # NOTE: in torchvision 0.22, Imagenette did not check if the download directory already exists, so we have to do it
    dataset_subfolder = os.path.join(save_path, 'imagenette2-320')
    train_folder = os.path.join(dataset_subfolder, 'train')
    val_folder = os.path.join(dataset_subfolder, 'val')

    download_data = not (os.path.isdir(train_folder) and os.path.isdir(val_folder))

    # Load test data with basic transformations
    # NOTE: we don't have to download the test set extra
    # as it is already downloaded with the training set
    train_dataset = datasets.Imagenette(
        root=save_path,
        split='train',
        size = '320px',
        download=download_data,
        transform = train_compose,
        )

    test_dataset = datasets.Imagenette(
        root=save_path, 
        split='val', 
        size = '320px', 
        download=False,
        transform = test_compose, 
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

    # Imagenette class names
    class_names = ['Tench','English springer','Cassette player',
                   'Chain saw','Church','French horn',
                   'Garbage truck','Gas pump','Golf ball',
                   'Parachute']

    return train_loader, test_loader, class_names


class PascalVocDataset(Dataset):
    def __init__(self, set_type, class_list, num_anchors=3, grid_size=7, image_size=(448,448), normalize=False, augment=False):
    
        assert set_type in {'train', 'val', 'test'}
        self.normalize = normalize
        self.augment = augment
        self.classes = class_list
        self.C = len(class_list)
        self.B = num_anchors
        self.S = grid_size
        
        self.image_size = image_size
    
        self.dataset = VOCDetection(
            root='../Dataset/PascalVOC',
            year='2012',
            image_set=('train' if set_type == 'train' else 'val'),
            download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Resize(size=(self.image_size))
            ])
        )
    
        # Generate class index if needed
        index = 0
        if len(self.classes) == 0:
            for i, data_pair in enumerate(tqdm(self.dataset, desc=f'Generating class dict')):
                data, label = data_pair
                for j, bbox_pair in enumerate(self.get_bounding_boxes(label)):
                    name, coords = bbox_pair
                    if name not in self.classes:
                        self.classes[name] = index
                        index += 1
            utils.save_class_dict(self.classes)

    def get_bounding_boxes(self, label):
        img_size = label['annotation']['size']
        width, height = int(img_size['width']), int(img_size['height'])
        x_scale = self.image_size[0] / width
        y_scale = self.image_size[1] / height
        boxes = []
        objects = label['annotation']['object']
        for obj in objects:
            box = obj['bndbox']
            coords = (
                int(int(box['xmin']) * x_scale),
                int(int(box['xmax']) * x_scale),
                int(int(box['ymin']) * y_scale),
                int(int(box['ymax']) * y_scale)
            )
            name = obj['name']
            boxes.append((name, coords))
        return boxes

    def scale_bbox_coord(self, coord, center, scale):
        return ((coord - center) * scale) + center

    def __getitem__(self, i):
        data, label = self.dataset[i]
        original_data = data
        x_shift = int((0.2 * random.random() - 0.1) * self.image_size[0])
        y_shift = int((0.2 * random.random() - 0.1) * self.image_size[1])
        scale = 1 + 0.2 * random.random()

        # Augment images
        if self.augment:
            data = TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            data = TF.adjust_hue(data, 0.2 * random.random() - 0.1)
            data = TF.adjust_saturation(data, 0.2 * random.random() + 0.9)
        if self.normalize:
            data = TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        grid_size_x = data.size(dim=2) / self.S  # Images in PyTorch have size (channels, height, width)
        grid_size_y = data.size(dim=1) / self.S

        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
        boxes = {}
        class_names = {}                    # Track what class each grid cell has been assigned to
        depth = 5 * self.B + self.C     # 5 numbers per bbox, then one-hot encoding of label
        ground_truth = torch.zeros((self.S, self.S, depth))
        for j, bbox_pair in enumerate(self.get_bounding_boxes(label)):
            name, coords = bbox_pair      
            assert name in self.classes, f"Unrecognized class '{name}'"
            class_index = self.classes.index(name)
            x_min, x_max, y_min, y_max = coords

            # Augment labels
            if self.augment:
                half_width = self.image_size[0] / 2
                half_height = self.image_size[1] / 2
                x_min = self.scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = self.scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = self.scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = self.scale_bbox_coord(y_max, half_height, scale) + y_shift

            # Calculate the position of center of bounding box
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)

            if 0 <= col < self.S and 0 <= row < self.S:
                cell = (row, col)
                if cell not in class_names or name == class_names[cell]:
                    # Insert class one-hot encoding into ground truth
                    one_hot = torch.zeros(self.C)
                    one_hot[class_index] = 1.0
                    ground_truth[row, col, :self.C] = one_hot
                    class_names[cell] = name

                    # Insert bounding box into ground truth tensor
                    bbox_index = boxes.get(cell, 0)
                    if bbox_index < self.B:
                        bbox_truth = (
                            (mid_x - col * grid_size_x) / self.image_size[0],     # X coord relative to grid square
                            (mid_y - row * grid_size_y) / self.image_size[1],     # Y coord relative to grid square
                            (x_max - x_min) / self.image_size[0],                 # Width
                            (y_max - y_min) / self.image_size[1],                 # Height
                            1.0                                                     # Confidence
                        )

                        # Fill all bbox slots with current bbox (starting from current bbox slot, avoid overriding prev)
                        # This prevents having "dead" boxes (zeros) at the end, which messes up IOU loss calculations
                        bbox_start = 5 * bbox_index + self.C
                        ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(self.B - bbox_index)
                        boxes[cell] = bbox_index + 1

        return data, ground_truth, original_data

    def __len__(self):
        return len(self.dataset)


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

