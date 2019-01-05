import abc
import os
import sys
import tqdm
import torch

from torch.utils.data import DataLoader
from typing import Callable, Any
from cs236605.train_results import BatchResult, EpochResult, FitResult


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """
    def __init__(self, model, loss_fn, optimizer, device=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        if self.device:
            model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs-1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

            # TODO: Train & evaluate for one epoch
            # - Use the train/test_epoch methods.
            # - Save losses and accuracies in the lists above.
            # - Optional: Implement checkpoints. You can use torch.save() to
            #   save the model to a file.
            # - Optional: Implement early stopping. This is a very useful and
            #   simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======
            # Start with training the model for 1 epoch & recording the results
            train_res = self.train_epoch(dl_train=dl_train)
            train_loss += train_res.losses
            train_acc.append(train_res.accuracy)
            
            # Run an evaluation of the model & save the results
            test_res = self.test_epoch(dl_test=dl_test)
            test_loss += test_res.losses
            test_acc.append(test_res.accuracy)
            
            # if torch.isnan(test_loss[-1]).item():
            #     print("Loss is NaN.\nBreaking training.")
            #     break

            if test_loss[-1] != test_loss[-1]:  # check if last loss is NaN
                print("Loss is NaN.\nBreaking training.")
                break
            
            if epoch == 0:
                best_acc = test_res.accuracy
                epochs_without_improvement = 0
                save_checkpoint = 1
                
            else:
                if test_res.accuracy > best_acc:
                    best_acc = test_res.accuracy
                    save_checkpoint = 1
                    epochs_without_improvement = 0

                else:
                    save_checkpoint = 0
                    epochs_without_improvement += 1
            
            # Create a checkpoint after each epoch
            if checkpoints is not None:
                if save_checkpoint:
                    file = open(checkpoints, 'wb')
                    torch.save(obj=self.model, f=file, pickle_protocol=3)
                    file.close()

            # Implement early stopping
            if early_stopping is not None:
                if epochs_without_improvement == early_stopping:
                    # We haven't improved at all in the last 'early_stopping' epochs
                    print("Reached the Early Stop condition.\nStopping the training.")
                    break
            # ========================

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)


class BlocksTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer):
        super().__init__(model, loss_fn, optimizer)

    def train_batch(self, batch) -> BatchResult:
        X, y = batch

        # TODO: Train the Block model on one batch of data.
        # - Forward pass
        # - Backward pass
        # - Optimize params
        # - Calculate number of correct predictions
        # ====== YOUR CODE: ======
        # We'll start with the forward pass, which will also give us our predictions
        y_pred = self.model.forward(X)
        
        # Now we can compute the loss with respect to the true labels
        loss = self.loss_fn.forward(y_pred, y)
        
        # Now we can compute the gradients with respect to the loss
        loss_grad = self.loss_fn.backward(loss)
        grads = self.model.backward(loss_grad)
        
        # And finally perform the optimization step
        self.optimizer.step()
        
        # All that's left is to compute the numebr of correct classification       
        model_pred = torch.argmax(input=y_pred, dim=1)
        incorrect_classifications = torch.nonzero(input=(model_pred - y))
        num_correct = y.shape[0] - incorrect_classifications.shape[0]
        # ========================

        return BatchResult(loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch

        # TODO: Evaluate the Block model on one batch of data.
        # - Forward pass
        # - Calculate number of correct predictions
        # ====== YOUR CODE: ======
        # We'll start with the forward pass, which will also give us our predictions
        y_pred = self.model.forward(X)
        
        # Now we can compute the loss with respect to the true labels
        loss = self.loss_fn.forward(y_pred, y)
        
        # All that's left is to compute the numebr of correct classification
        model_pred = torch.argmax(y_pred, dim=1)
        incorrect_classifications = torch.nonzero(input=(model_pred - y))
        num_correct = y.shape[0] - incorrect_classifications.shape[0]     
        # ========================

        return BatchResult(loss, num_correct)


class TorchTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)

    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        # TODO: Train the PyTorch model on one batch of data.
        # - Forward pass
        # - Backward pass
        # - Optimize params
        # - Calculate number of correct predictions
        # ====== YOUR CODE: ======
        # We'll start with the forward pass, which will also give us our predictions
        y_pred = self.model.forward(X)
        
        # Zero the gradients after each step
        self.optimizer.zero_grad()

        # Compute the loss with respect to the true labels
        loss = self.loss_fn(y_pred, y)
        
        # Run the backwards pass
        loss.backward()

        loss = loss.data.tolist()
        # Perform the optimization step
        self.optimizer.step()
        
        # All that's left is to compute the numebr of correct classification       
        model_pred = torch.argmax(input=y_pred, dim=1)
        incorrect_classifications = torch.nonzero(input=(model_pred - y))
        num_correct = y.shape[0] - incorrect_classifications.shape[0]
        # ========================

        return BatchResult(loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            # TODO: Evaluate the PyTorch model on one batch of data.
            # - Forward pass
            # - Calculate number of correct predictions
            # ====== YOUR CODE: ======
            # We'll start with the forward pass, which will also give us our predictions
            y_pred = self.model.forward(X)

            # Now we can compute the loss with respect to the true labels
            loss = self.loss_fn.forward(y_pred, y)
            loss = loss.data.tolist()

            # All that's left is to compute the numebr of correct classification
            model_pred = torch.argmax(y_pred, dim=1)
            incorrect_classifications = torch.nonzero(input=(model_pred - y))
            num_correct = y.shape[0] - incorrect_classifications.shape[0]    
            # ========================

        return BatchResult(loss, num_correct)
