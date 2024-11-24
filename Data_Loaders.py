import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset

class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')

        # Separate majority and minority classes
        class_0 = self.data[self.data[:, -1] == 0]  # Majority class
        class_1 = self.data[self.data[:, -1] == 1]  # Minority class

        # Balance the dataset by undersampling majority class
        n = len(class_1) * 4
        undersampled_class_0 = class_0[np.random.choice(class_0.shape[0], n, replace=False)]
        balanced_data = np.vstack((undersampled_class_0, class_1))
        np.random.shuffle(balanced_data)

        # Normalize balanced data
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(balanced_data)
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb"))

    def __len__(self):
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()

        x = torch.tensor(self.normalized_data[idx, :-1], dtype=torch.float32)
        y = torch.tensor(self.normalized_data[idx, -1], dtype=torch.float32)
        return {'input': x, 'label': y}


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()

        # Get dataset size
        dataset_size = len(self.nav_dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        # Split indices for training and testing (70/30 split)
        split = int(0.7 * dataset_size)
        train_indices, test_indices = indices[:split], indices[split:]

        # Create PyTorch Subset for train and test sets
        train_subset = Subset(self.nav_dataset, train_indices)
        test_subset = Subset(self.nav_dataset, test_indices)

        # Create DataLoader for train and test sets
        self.train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)

    # Iterate over train_loader and test_loader
    for idx, sample in enumerate(data_loaders.train_loader):
        input_data, label_data = sample['input'], sample['label']
        print(f"Train Sample {idx}: Input: {input_data}, Label: {label_data}")

    for idx, sample in enumerate(data_loaders.test_loader):
        input_data, label_data = sample['input'], sample['label']
        print(f"Test Sample {idx}: Input: {input_data}, Label: {label_data}")


if __name__ == '__main__':
    main()
