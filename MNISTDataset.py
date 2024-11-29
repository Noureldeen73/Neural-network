from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt


class MNISTDataset(Dataset):
    def __init__(self, image_data_root, label_data_root):
        #Image set variables
        self.image_data_root = image_data_root
        self.image_magic_number = 0
        self.image_number_of_images = 0
        self.image_number_of_rows = 0
        self.image_number_of_columns = 0
        self.images = np.empty(0)

        #Label set variables
        self.label_data_root = label_data_root
        self.label_magic_number = 0
        self.label_number_of_labels = 0
        self.labels = np.empty(0)

        #Load data
        self.image_init_dataset()
        self.label_init_dataset()

    def __len__(self):
        return self.image_number_of_images

    def __getitem__(self, idx):
        image = self.images[idx].reshape(1, 28, 28)
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def image_init_dataset(self):
        image_file = open(self.image_data_root, "rb")
        reorder_type = np.dtype(np.uint32).newbyteorder('>')

        self.image_magic_number = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.image_number_of_images = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.image_number_of_rows = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]
        self.image_number_of_columns = np.frombuffer(image_file.read(4), dtype=reorder_type)[0]

        buffer = image_file.read(self.image_number_of_images * self.image_number_of_rows * self.image_number_of_columns)
        self.images = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
        self.images = np.reshape(self.images, (self.image_number_of_images , 784))
        self.images = self.images/255.0
        self.images = torch.tensor(self.images)
    def label_init_dataset(self):
        label_file = open(self.label_data_root, "rb")
        reorder_type = np.dtype(np.uint32).newbyteorder('>')

        self.label_magic_number = np.frombuffer(label_file.read(4), dtype=reorder_type)[0]
        self.label_number_of_labels = np.frombuffer(label_file.read(4), dtype=reorder_type)[0]

        buffer = label_file.read(self.label_number_of_labels)

        self.labels = np.frombuffer(buffer, dtype = np.uint8)

        self.labels = torch.tensor(self.labels, dtype= torch.long)