import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import cs236605.dataloader_utils as dataloader_utils
from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train, y_train = dataloader_utils.flatten(dl_train)
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = len(set(y_train.numpy()))
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = self.calc_distances(x_test)

        # TODO: Implement k-NN class prediction based on distance matrix.
        # For each training sample we'll look for it's k-nearest neighbors.
        # Then we'll predict the label of that sample to be the majority
        # label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)

        for i in range(n_test):
            # TODO:
            # - Find indices of k-nearest neighbors of test sample i
            # - Set y_pred[i] to the most common class among them

            # ====== YOUR CODE: ======
            # Start by extracting the distances for each sample in x_test
            dists = dist_matrix[:, i]
            
            # Find k-nearest neighbors
#             sorted_dists, sorted_inds = torch.sort(dists, descending=False)
#             k_inds = sorted_inds[0:self.k]

            ktop_inds = torch.topk(input=dists, k=self.k, largest=False)
            k_inds = ktop_inds[1]
            
            # Find the most frequent label
            k_labels = [self.y_train[ind] for ind in k_inds]
            counts = np.bincount(k_labels)
            y_pred[i] = torch.Tensor([np.argmax(counts), ]) 
            # ========================

        return y_pred

    def calc_distances(self, x_test: Tensor):
        """
        Calculates the L2 distance between each point in the given test
        samples to each point in the training samples.
        :param x_test: Test samples. Should be a tensor of shape (Ntest,D).
        :return: A distance matrix of shape (Ntrain,Ntest) where Ntrain is the
            number of training samples. The entry i, j represents the distance
            between training sample i and test sample j.
        """

        # TODO: Implement L2-distance calculation as efficiently as possible.
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - No credit will be given for an implementation with two explicit
        #   loops.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops). Hint: Open the expression (a-b)^2.

        dists = torch.tensor([])
        # ====== YOUR CODE: ======
        # Get the dimensions of each matrix
        train_shape = list(self.x_train.size())
        test_shape = list(x_test.size())
        
        # Calculate the joint terms
        addition_term = -2 * torch.mm(self.x_train, torch.t(x_test))
        
        # Calculate the square terms
        train = torch.pow(self.x_train, 2)
        test = torch.pow(x_test, 2)
        
        ones = torch.ones((train_shape[1], test_shape[0]))
        train = torch.mm(train, ones)
        
        ones = torch.ones((test_shape[1], train_shape[0]))
        test = torch.t(torch.mm(test, ones))
        
        dists = train + test + addition_term
        # ========================
        
        return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.

    accuracy = None
    # ====== YOUR CODE: ======
    acc = y - y_pred
    ones = torch.ones(y.shape, dtype=torch.long)
    acc = torch.where(acc == 0, acc, ones)
    s = torch.sum(acc).data.numpy()
    accuracy = 1 - s / y.shape[0]
    
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []
    
    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO: Train model num_folds times with different train/val data.
        # Don't use any third-party libraries.
        # You can use your train/validation splitter from part 1 (even if
        # that means that it's not really k-fold CV since it will be a
        # different split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        # Generate the starting and ending indices for each fold - In order to simplify the scanning process        
        n_samples = ds_train.subset_len        
        fold_size = n_samples // num_folds
        validation_ratio = fold_size / n_samples
        
        fold_starting_inds = np.array([j * fold_size for j in range(num_folds)])
                
        acc = []
        for j in range(num_folds):
            # Seperate the training & validation folds
            if j == 0:
                train_inds = np.arange(fold_size, n_samples)
                valid_inds = np.arange(fold_size)
        
            elif j == num_folds - 1:
                train_inds = np.arange((n_samples - fold_size))
                valid_inds = np.arange((n_samples - fold_size), n_samples)
                
            else:                
                train_inds_1 = np.arange(0, fold_starting_inds[j])                
                train_inds_2 = np.arange((fold_starting_inds[j] + fold_size), n_samples)
                                                
                train_inds = np.concatenate((train_inds_1, train_inds_2))
                valid_inds = np.arange(fold_starting_inds[j], (fold_starting_inds[j] + fold_size))
            
            train_samp = SubsetRandomSampler(train_inds.tolist())
            valid_samp = SubsetRandomSampler(valid_inds.tolist())

            dl_train = torch.utils.data.DataLoader(ds_train, batch_size=100, shuffle=False, num_workers=1,
                                                   sampler=train_samp)
                                       
            dl_valid = torch.utils.data.DataLoader(ds_train, batch_size=100, shuffle=False, num_workers=1,
                                                   sampler=valid_samp)
                                       
            # Train & calculate the accuracy on the current folds division
            knn_classifier = KNNClassifier(k=k)
            knn_classifier.train(dl_train)

            x_valid, y_valid = dataloader_utils.flatten(dl_valid)
            y_pred = knn_classifier.predict(x_valid)

            # Calculate accuracy
            tmp_acc = accuracy(y_valid, y_pred)
            
            acc.append(tmp_acc)
        
        accuracies.append(acc)
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
