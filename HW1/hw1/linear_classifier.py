import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple
from .losses import ClassifierLoss


class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO: Create weights tensor of appropriate dimensions
        # Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        dist = torch.distributions.normal.Normal(loc=0, scale=weight_std)
        self.weights = dist.sample(sample_shape=(n_features, n_classes))
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO: Implement linear prediction.
        # Calculate the score for each class using the weights and
        # return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.matmul(x, self.weights)
        y_pred = torch.argmax(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO: calculate accuracy of prediction.
        # Use the predict function above and compare the predicted class
        # labels to the ground truth labels to obtain the accuracy (in %).
        # Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        shape = y_pred.shape
        N = shape[0]
        diff = y_pred != y
        diff_sum = torch.matmul(torch.ones_like(y_pred), diff.type(y.type()))
        acc = 1 - float(diff_sum) / float(N)
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):

            # TODO: Implement model training loop.
            # At each epoch, evaluate the model on the entire training set
            # (batch by batch) and update the weights.
            # Each epoch, also evaluate on the validation set.
            # Accumulate average loss and total accuracy for both sets.
            # The train/valid_res variables should hold the average loss and
            # accuracy per epoch.
            #
            # Don't forget to add a regularization term to the loss, using the
            # weight_decay parameter.

            total_correct = 0
            average_loss = 0

            # ====== YOUR CODE: ======
            # Define
            self._last_weights = None
            self._best_acc = 0
            # epsilon = 0.76  # the drop we permit in accuracy (in percents %)
            # epsilon = 1
            bad_epochs_counter = 0
            max_bad_epochs = 3

            # helper method definition
            def train_validate_epoch(dl, mode='Train', average_loss=0, total_correct=0):
                loss = []
                correct = []
                batch = None

                for batch in dl:
                    x = batch[0]
                    y = batch[1]
                    y_predicted, x_scores = self.predict(x)
                    # average_loss += loss_fn.loss(x, y, x_scores, y_predicted) + weight_decay
                    loss.append(loss_fn.loss(x, y, x_scores, y_predicted) + weight_decay)
                    grad = loss_fn.grad()
                    # total_correct += self.evaluate_accuracy(y, y_predicted)
                    correct.append(self.evaluate_accuracy(y, y_predicted))

                    # update weights - only when training
                    if mode == 'Train':
                        self.weights = self.weights - learn_rate * grad

                # evaluate last batch
                y_predicted, x_scores = self.predict(batch[0])
                loss.append(loss_fn.loss(batch[0], batch[1], x_scores, y_predicted))
                correct.append(self.evaluate_accuracy(batch[1], y_predicted))

                # Calculate average loss and correct
                average_loss = sum(loss)/len(loss)
                total_correct = sum(correct)/len(correct)

                if mode == 'Train':
                    train_res.accuracy.append(total_correct)
                    train_res.loss.append(average_loss)
                elif mode == 'Validation':
                    valid_res.accuracy.append(total_correct)
                    valid_res.loss.append(average_loss)
                else:
                    raise ValueError

                return self.weights

            # Train batch by batch the whole epoch
            last_weights = train_validate_epoch(dl_train, mode='Train', average_loss=average_loss,
                                                total_correct=total_correct)

            # Validate batch by batch the whole epoch
            train_validate_epoch(dl_valid, mode='Validation', average_loss=average_loss, total_correct=total_correct)

            # Stop training when maximal number of bad epochs passed or when the diff is too bad 
            if len(valid_res.accuracy) >= 2:
                acc_diff = valid_res.accuracy[-1] - valid_res.accuracy[-2]
                if acc_diff > 0:
                    bad_epochs_counter = 0
                    if valid_res.accuracy[-1] > self._best_acc:
                        self._best_acc = valid_res.accuracy[-1]
                        self._last_weights = last_weights
                else:
                    bad_epochs_counter = bad_epochs_counter + 1
                    if bad_epochs_counter > max_bad_epochs:
                        if self._last_weights is not None:
                            self.weights = self._last_weights
                        break
            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be at the end).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO: Convert the weights matrix into a tensor of images.
        # The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        shape = tuple([self.n_classes]) + img_shape
        if has_bias:
            w_images = torch.reshape(self.weights[:-1], shape)
        else:
            w_images = torch.reshape(self.weights, shape)
        # ========================

        return w_images
