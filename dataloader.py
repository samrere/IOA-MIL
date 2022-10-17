"""Pytorch dataset object that loads MNIST dataset as bags."""
import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        pre_load=False
        if self.train:
            try:
                all_imgs=torch.load('./datasets/train_all_imgs.pt')
                all_labels=torch.load('./datasets/train_all_labels.pt')
                pre_load=True
            except FileNotFoundError:
                loader = data_utils.DataLoader(datasets.MNIST('./datasets',
                                                              train=True,
                                                              download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,), (0.3081,))])),
                                               batch_size=self.num_in_train,
                                               shuffle=False)
        else:
            try:
                all_imgs=torch.load('./datasets/test_all_imgs.pt')
                all_labels=torch.load('./datasets/test_all_labels.pt')
                pre_load=True
            except FileNotFoundError:
                loader = data_utils.DataLoader(datasets.MNIST('./datasets',
                                                              train=False,
                                                              download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,), (0.3081,))])),
                                               batch_size=self.num_in_test,
                                               shuffle=False)
        if not pre_load:
            for (batch_data, batch_labels) in loader:
                all_imgs = batch_data
                all_labels = batch_labels
                torch.save(all_imgs, f"datasets/{'train' if self.train else 'test'}_all_imgs.pt")
                torch.save(all_labels, f"datasets/{'train' if self.train else 'test'}_all_labels.pt")

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label

