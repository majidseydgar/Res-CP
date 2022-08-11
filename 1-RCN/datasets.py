import os.path
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import scipy.io as sio
from sklearn import preprocessing

class Dataset_call(object):
    def __init__(self, dataset_name, root, patch_length=4, return_zero_labels=False, sample_num=10,
                 batch_size=16):

        if dataset_name == 'UP':
            data_dir = os.path.join(root, 'UP/')
            data = sio.loadmat(data_dir + "PaviaU.mat")
            label = sio.loadmat(data_dir + 'PaviaU_gt.mat')
            self.data = data["paviaU"]
            self.label = label["paviaU_gt"]
        if dataset_name == "SA":
            data_dir = os.path.join(root, 'SA/')
            data = sio.loadmat(data_dir + "Salinas_corrected.mat")
            label = sio.loadmat(data_dir + 'Salinas_gt.mat')
            self.data = data["salinas_corrected"]
            self.label = label["salinas_gt"]
        if dataset_name == "KSC":
            data_dir = os.path.join(root, 'KSC/')
            data = sio.loadmat(data_dir + "KSC.mat")
            label = sio.loadmat(data_dir + 'KSC_gt.mat')
            self.data = data["KSC"]
            self.label = label["KSC_gt"]
        if dataset_name == "UH":
            data_dir = os.path.join(root, 'UH/')
            data = sio.loadmat(data_dir + "Houston.mat")
            label = sio.loadmat(data_dir + 'Houston_gt.mat')
            self.data = data["Houston"]
            self.label = label["Houston_gt"]

        self.Input_dimension = self.data.shape[2]
        self.PATCH_LENGTH = patch_length
        self.gt = self.label.reshape(np.prod(self.label.shape[:2]), )
        self.batch_size = batch_size
        self.padded_data = self.pre_processing()
        self.sample_num = sample_num
        self.train_indices, self.test_indices = self.sample_spatial()
        self.total_indices = [j for j, x in enumerate(self.gt.ravel().tolist())]
        self.train_size = len(self.train_indices)
        self.test_size = len(self.test_indices)
        self.Total_size = self.data.shape[0] * self.data.shape[1]
        self.num_classes = max(self.gt)
        self.return_zero_labels = return_zero_labels
        self.dataset_height = self.data.shape[0]
        self.dataset_width = self.data.shape[1]
        self.y_train = self.gt[self.train_indices] - 1
        self.y_test = self.gt[self.test_indices] - 1


    def pre_processing(self):

        data_ = self.data
        self.data = self.data.reshape(np.prod(self.data.shape[:2]), np.prod(self.data.shape[2:]))
        self.data = preprocessing.scale(self.data)
        self.data = self.data.reshape(data_.shape[0], data_.shape[1], data_.shape[2])
        padded_data = np.lib.pad(
            self.data, ((self.PATCH_LENGTH, self.PATCH_LENGTH), (self.PATCH_LENGTH, self.PATCH_LENGTH),
                        (0, 0)), 'constant', constant_values=0)
        return padded_data

    def sample_spatial(self):
        # seeds = [4231]
        # np.random.seed(seeds[0])
        sample_num = self.sample_num
        groundTruth = self.label
        New_train = np.zeros_like(groundTruth)
        New_test = groundTruth
        class_size = np.max(groundTruth)
        for i in range(class_size):
            idx = np.column_stack(np.where(groundTruth == i + 1))
            T_idx = idx[np.random.choice(idx.shape[0], sample_num, replace=False), :]
            for id1 in range(sample_num):
                New_train[T_idx[id1, 0], T_idx[id1, 1]] = i + 1
                New_test[T_idx[id1, 0], T_idx[id1, 1]] = 0
        return [j for j, x in enumerate(New_train.ravel().tolist())if x >0], \
               [j for j, x in enumerate(New_test.ravel().tolist())if x >0]

    def sampling(self):
        proportion = self.sample_num
        ground_truth = self.gt
        train = {}
        test = {}
        labels_loc = {}
        m = max(ground_truth)
        for i in range(m):
            indexes = [
                j for j, x in enumerate(ground_truth.ravel().tolist())
                if x == i + 1
            ]
            np.random.shuffle(indexes)
            labels_loc[i] = indexes
            if proportion > 1:
                nb_val = proportion
            elif proportion != 1:
                nb_val = max(int((1 - proportion) * len(indexes)), 3)
            else:
                nb_val = 0
            train[i] = indexes[:nb_val]
            test[i] = indexes[nb_val:]
        train_indexes = []
        test_indexes = []
        for i in range(m):
            train_indexes += train[i]
            test_indexes += test[i]
        np.random.shuffle(train_indexes)
        np.random.shuffle(test_indexes)
        # print("Total number of training samples: ", len(train_indexes))
        # print("Total number of test samples : ", len(test_indexes))
        return train_indexes, test_indexes

    def index_assignment(self, index, row, col, pad_length):
        new_assign = {}
        for counter, value in enumerate(index):
            assign_0 = value // col + pad_length
            assign_1 = value % col + pad_length
            new_assign[counter] = [assign_0, assign_1]
        return new_assign

    def select_patch(self, matrix, pos_row, pos_col, ex_len):
        # print(pos_row, pos_col, ex_len)
        selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1)]
        selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
        return selected_patch

    def select_small_cubic(self, data_size, data_indices, whole_data, patch_length, padded_data, dimension):
        small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
        data_assign = self.index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
        for i in range(len(data_assign)):
            small_cubic_data[i] = self.select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
        return small_cubic_data


    def generate_iter(self):
        gt_all = self.gt - 1

        train_data = self.select_small_cubic(len(self.train_indices), self.train_indices, self.data,
                                        self.PATCH_LENGTH, self.padded_data, self.Input_dimension)

        test_data = self.select_small_cubic(self.test_size, self.test_indices, self.data,
                                       self.PATCH_LENGTH, self.padded_data, self.Input_dimension)
        x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], self.Input_dimension)
        x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], self.Input_dimension)

        #
        x_test = x_test_all
        # y_train = self.y_train
        # y_test = self.y_test

        x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
        y1_tensor_train = torch.from_numpy(self.y_train).type(torch.FloatTensor)
        torch_dataset_train = TensorDataset(x1_tensor_train, y1_tensor_train)
        #

        x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
        y1_tensor_test = torch.from_numpy(self.y_test).type(torch.FloatTensor)
        torch_dataset_test = TensorDataset(x1_tensor_test, y1_tensor_test)

        train_iter = DataLoader(
            dataset=torch_dataset_train,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,
            num_workers=0,
        )

        test_iter = DataLoader(
            dataset=torch_dataset_test,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=False,
            num_workers=0,
        )


        if self.return_zero_labels == True:
            all_data = self.select_small_cubic(self.Total_size, self.total_indices, self.data,
                                          self.PATCH_LENGTH, self.padded_data, self.Input_dimension)

            all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], self.Input_dimension)
            all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
            all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
            torch_dataset_all = TensorDataset(all_tensor_data, all_tensor_data_label)

            all_iter = DataLoader(
                dataset=torch_dataset_all,  # torch TensorDataset format
                batch_size=self.batch_size,  # mini batch size
                shuffle=False,
                num_workers=0,
            )


            return train_iter, test_iter, all_iter  # , valiada_iter # , y_test

        else:
            return train_iter, test_iter


    def get_iterator(self):

        TRAIN_SIZE = len(self.train_indices)
        # TEST_SIZE = self.Total_size - TRAIN_SIZE
        # whole_data = self.data
        #
        if self.return_zero_labels == True:
            train_iter, test_iter, all_iter = self.generate_iter()  # batchsize in 1
            return train_iter, test_iter, all_iter
        else:
            train_iter, test_iter = self.generate_iter()
            return train_iter, test_iter