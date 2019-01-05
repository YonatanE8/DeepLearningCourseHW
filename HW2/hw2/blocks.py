import abc
import torch
import numpy as np


class Block(abc.ABC):
    """
    A block is some computation element in a network architecture which
    supports automatic differentiation using forward and backward functions.
    """
    def __init__(self):
        # Store intermediate values needed to compute gradients in this hash
        self.grad_cache = {}
        self.training_mode = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Computes the forward pass of the block.
        :param args: The computation arguments (implementation specific).
        :return: The result of the computation.
        """
        pass

    @abc.abstractmethod
    def backward(self, dout):
        """
        Computes the backward pass of the block, i.e. the gradient
        calculation of the final network output with respect to each of the
        parameters of the forward function.
        :param dout: The gradient of the network with respect to the
        output of this block.
        :return: A tuple with the same number of elements as the parameters of
        the forward function. Each element will be the gradient of the
        network output with respect to that parameter.
        """
        pass

    @abc.abstractmethod
    def params(self):
        """
        :return: Block's trainable parameters and their gradients as a list
        of tuples, each tuple containing a tensor and it's corresponding
        gradient tensor.
        """
        pass

    def train(self, training_mode=True):
        """
        Changes the mode of this block between training and evaluation (test)
        mode. Some blocks have different behaviour depending on mode.
        :param training_mode: True: set the model in training mode. False: set
        evaluation mode.
        """
        self.training_mode = training_mode


class Linear(Block):
    """
    Fully-connected linear layer.
    """

    def __init__(self, in_features, out_features, wstd=0.1):
        """
        :param in_features: Number of input features (Din)
        :param out_features: Number of output features (Dout)
        :wstd: standard deviation of the initial weights matrix
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # TODO: Create the weight matrix (w) and bias vector (b).
        # ====== YOUR CODE: ======
        # Initialize the weights matrix using a normal distribution with the given std parameter
        w = torch.empty(out_features, in_features)       
        self.w = w.normal_(mean=0, std=wstd)
        
        # Initialize the bias vector using a normal distribution with the given std parameter
        self.b = torch.zeros([out_features, ])
        # ========================

        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def params(self):
        return [
            (self.w, self.dw), (self.b, self.db)
        ]

    def forward(self, x, **kw):
        """
        Computes an affine transform, y = x W^T + b.
        :param x: Input tensor of shape (N,Din) where N is the batch
        dimension, and Din is the number of input features, or of shape
        (N,d1,d2,...,dN) where Din = d1*d2*...*dN.
        :return: Affine transform of each sample in x.
        """

        x = x.reshape((x.shape[0], -1))

        # TODO: Compute the affine transform

        # ====== YOUR CODE: ======        
        # Just a basic matrix multiplication
        out = torch.mm(x, torch.transpose(input=self.w, dim0=0, dim1=1)) + self.b 
        # ========================

        self.grad_cache['x'] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, Dout).
        :return: Gradient with respect to block input, shape (N, Din)
        """
        x = self.grad_cache['x']

        # TODO: Compute
        #   - dx, the gradient of the loss with respect to x
        #   - dw, the gradient of the loss with respect to w
        #   - db, the gradient of the loss with respect to b
        # You should accumulate gradients in dw and db.
        # ====== YOUR CODE: ======
        # Compute the the derviatives according to the following definitions:
        # dx = d(L) / d(out) * d(out) / d(x)
        # d(out) / d(x) = W^T
        # dw = d(L) / d(out) * d(out) / d(w)
        # dw = d(out) / d(w) = x
        # db = d(L) / d(out) * d(out) / d(b)
        # db = 1
        dx = torch.mm(dout, self.w)
        dw = torch.mm(torch.transpose(x, 0, 1), dout)
        db = torch.squeeze(torch.mm(torch.ones([1, int(dout.shape[0])]), dout), dim=0)
        
        self.dw += torch.transpose(dw, dim0=0, dim1=1)
        self.db += db
        # ========================

        return dx

    def __repr__(self):
        return f'Linear({self.in_features}, {self.out_features})'


class ReLU(Block):
    """
    Rectified linear unit.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes max(0, x).
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: ReLU of each sample in x.
        """

        # TODO: Implement the ReLU operation.
        # ====== YOUR CODE: ======
        # Create a mask to multiply x with.
        o = torch.ones_like(input=x, dtype=x.dtype, requires_grad=False)
        z = torch.zeros_like(input=x, dtype=x.dtype, requires_grad=False)
        mask = torch.where(x > 0, o, z)
        
        # Perform the ReLU operation
        out = x * mask
        # ========================

        self.grad_cache['x'] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to block output, shape (N, *).
        :return: Gradient with respect to block input, shape (N, *)
        """
        x = self.grad_cache['x']

        # TODO: Implement gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        # Define a constant tensor of {1, 0}
        o = torch.ones_like(input=x, dtype=x.dtype, requires_grad=False)
        z = torch.zeros_like(input=x, dtype=x.dtype, requires_grad=False)
        
        # Take 1 ehere x[i] > 0, and 0 otherwise
        mask = torch.where(x > 0, o, z)
        dx = dout * mask
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return 'ReLU'


class CrossEntropyLoss(Block):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Computes cross-entropy loss directly from class scores.
        Given class scores x, and a 1-hot encoding of the correct class yh,
        the cross entropy loss is defined as: -yh^T * log(softmax(x)).

        This implementation works directly with class scores (x) and labels
        (y), not softmax outputs or 1-hot encodings.

        :param x: Tensor of shape (N,D) where N is the batch
        dimension, and D is the number of features. Should contain class
        scores, NOT PROBABILITIES.
        :param y: Tensor of shape (N,) containing the ground truth label of
        each sample.
        :return: Cross entropy loss, as if we computed the softmax of the
        scores, encoded y as 1-hot and calculated cross-entropy by
        definition above. A scalar.
        """

        N = x.shape[0]
        xmax, _ = torch.max(x, dim=1, keepdim=True)
        x = x - xmax  # for numerical stability

        # TODO: Compute the cross entropy loss using the last formula from the
        # notebook (i.e. directly using the class scores).
        # Tip: to get a different column from each row of a matrix tensor m,
        # you can index it with m[range(num_rows), list_of_cols].
        # ====== YOUR CODE: ======
        # Start by getting the correct X scores
        scores = x[range(x.shape[0]), y]
    
        # Compute the sums of the exponents
        sums = torch.sum(torch.exp(x), dim=1)
    
        # Get the final cross-entropy loss
        loss = -scores + torch.log(sums)
        loss = torch.mean(loss)
        # ========================

        self.grad_cache['x'] = x
        self.grad_cache['y'] = y
        return loss

    def backward(self, dout=1.0):
        """
        :param dout: Gradient with respect to block output, a scalar which
        defaults to 1 since the output of forward is scalar.
        :return: Gradient with respect to block input (only x), shape (N,D)
        """
        x = self.grad_cache['x']
        y = self.grad_cache['y']
        N = x.shape[0]
        
        # TODO: Calculate the gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        # The derivaitive the loss with respect to a single sample X is:
        # d(loss) / d(X_y) = -1 + exp(X_y) * (1 / (sum(exp(X))))
        # d(loss) / d(X_j) = exp(X_j) * (1 / (sum(exp(X))))
        # Take the exponent of all X values
        x_exp = torch.exp(x)
        
        # Take the relevant scores vector
        true_scores = x_exp[range(x.shape[0]), y]
        
        # Compute sums of the exponents
        sums = torch.sum(input=x_exp, dim=1)
        
        # Compute the final gradient  
        true_g = -1 + (true_scores / sums) 
        
        # Insert the gradient vector into an x-like matrix padded with 0 for all other inputs other then x_y
        dx = x_exp / sums.unsqueeze_(dim=1)
        dx[range(x.shape[0]), y] = true_g
        dx /= N
        dx *= dout
        # ========================

        return dx

    def params(self):
        return []


class Dropout(Block):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0. <= p <= 1.
        self.p = p

    def forward(self, x, **kw):
        # TODO: Implement the dropout forward pass. Notice that contrary to
        # previous blocks, this block behaves differently a according to the
        # current mode (train/test).
        # ====== YOUR CODE: ======
        # Check if we are at training mode
        if self.training_mode:
            probs = torch.ones_like(input=x[0, :], dtype=x.dtype, requires_grad=False) * self.p
            self.dropout_mask = torch.bernoulli(input=probs)
            self.dropout_mask = self.dropout_mask.repeat([x.shape[0], 1])
            out = self.dropout_mask * x 
            
        else:
            # We are at testing mode
            out = x 
        # ========================

        return out

    def backward(self, dout):
        # TODO: Implement the dropout backward pass.
        # ====== YOUR CODE: ======
        # Check if we are at training mode
        if self.training_mode:
            dx = dout * self.dropout_mask
            
        else:
            # We are at testing mode
            dx = dout
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return f'Dropout(p={self.p})'


class Sequential(Block):
    """
    A Block that passes input through a sequence of other blocks.
    """
    def __init__(self, *blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, **kw):
        out = None

        # TODO: Implement the forward pass by passing each block's output
        # as the input of the next.
        # ====== YOUR CODE: ======
        for i, block in enumerate(self.blocks):
            if i == 0:
                out = block.forward(x)
                
            elif isinstance(block, CrossEntropyLoss):
                out = block.forward(out, **kw)
                continue 
                
            else: 
                out = block.forward(out)
        # ========================

        return out

    def backward(self, dout):
        din = None

        # TODO: Implement the backward pass.
        # Each block's input gradient should be the previous block's output
        # gradient. Behold the backpropagation algorithm in action!
        # ====== YOUR CODE: ======
        first = 1
        inds = list(np.arange(len(self.blocks)))
        inds.reverse()
        for i in inds:
            if first:
                din = self.blocks[i].backward(dout)
                first = 0
            
            else:
                din = self.blocks[i].backward(din)
        # ========================

        return din

    def params(self):
        params = []

        # TODO: Return the parameter tuples from all blocks.
        # ====== YOUR CODE: ======
        for block in self.blocks:
            param = block.params()
            
            if len(param):
                params += param
        # ========================

        return params

    def train(self, training_mode=True):
        for block in self.blocks:
            block.train(training_mode)

    def __repr__(self):
        res = 'Sequential\n'
        for i, block in enumerate(self.blocks):
            res += f'\t[{i}] {block}\n'
        return res

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, item):
        return self.blocks[item]

