import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Dropout, Sequential
# from blocks import Block, Linear, ReLU, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is:

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.

    If dropout is used, a dropout layer is added after every ReLU.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param: Dropout probability. Zero means no dropout.
        """
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.dropout = dropout

        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        for i, h_features in enumerate(hidden_features):
            if i == 0:
                # Add the first layer
                blocks.append(Linear(in_features=in_features, out_features=hidden_features[0]))
                blocks.append(ReLU())
                
                if self.dropout != 0:
                    blocks.append(Dropout(p=self.dropout))
                
            else:
                blocks.append(Linear(in_features=hidden_features[i - 1], out_features=hidden_features[i]))
                blocks.append(ReLU())
                
                if self.dropout != 0:
                    blocks.append(Dropout(p=self.dropout))
                
        # Add the last layer
        blocks.append(Linear(in_features=hidden_features[-1], out_features=num_classes))
        # ========================

        self.sequence = Sequential(*blocks)
       
    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        if isinstance(self.filters[0], list):
            first = 1
            for i, filters in enumerate(self.filters):
                for j, filter_ in enumerate(filters):
                    if first:
                        # Create the first convolution layer
                        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=filter_, kernel_size=[3, 3],
                                                stride=1, padding=1, dilation=1, groups=1, bias=True))
                        layers.append(nn.ReLU())
                        first = 0

                    else:
                        if j == 0 and i != 0:
                            layers.append(nn.Conv2d(in_channels=self.filters[i-1][0], out_channels=filter_,
                                                    kernel_size=[3, 3], stride=1, padding=1, dilation=1, groups=1,
                                                    bias=True))
                            layers.append(nn.ReLU())

                        else:
                            layers.append(nn.Conv2d(in_channels=filters[j - 1], out_channels=filter_,
                                                    kernel_size=[3, 3], stride=1, padding=1, dilation=1, groups=1,
                                                    bias=True))
                            layers.append(nn.ReLU())

                if (i + 1) % self.pool_every == 0:
                    layers.append(nn.MaxPool2d(kernel_size=[2, 2], stride=None, padding=0, dilation=1))

        else:
            for i, filters in enumerate(self.filters):
                if i == 0:
                    # Create the first convolution layer
                    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=[3, 3],
                                            stride=1, padding=1, dilation=1, groups=1, bias=True))
                    layers.append(nn.ReLU())

                else:
                    layers.append(nn.Conv2d(in_channels=self.filters[i - 1], out_channels=filters, kernel_size=[3, 3],
                                            stride=1, padding=1, dilation=1, groups=1, bias=True))
                    layers.append(nn.ReLU())

                if (i + 1) % self.pool_every == 0:
                    layers.append(nn.MaxPool2d(kernel_size=[2, 2], stride=None, padding=0, dilation=1))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        num_of_layers = len(self.filters)
        num_of_pools = num_of_layers // self.pool_every

        if num_of_pools == 0:
            num_of_in_channels = int((self.in_size[1]) ** 2)

        else:
            num_of_in_channels = int((self.in_size[1] // (2 * num_of_pools)) ** 2)

        for i, dims in enumerate(self.hidden_dims):
            if i == 0:
                if isinstance(self.filters[-1], list):
                    f = self.filters[-1][0]

                else:
                    f = self.filters[-1]

                layers.append(nn.Linear(in_features=(f * num_of_in_channels), out_features=dims,
                                        bias=True))
                layers.append(nn.ReLU())
                
            else:
                layers.append(nn.Linear(in_features=self.hidden_dims[i - 1], out_features=dims, bias=True))
                layers.append(nn.ReLU())
                
        layers.append(nn.Linear(in_features=self.hidden_dims[-1], out_features=self.out_classes, bias=True))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        out = self.feature_extractor.forward(x)
        out = torch.reshape(out, (out.size()[0], -1))
        out = self.classifier.forward(out)
        # ========================
        
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        layers = []

        if isinstance(self.filters[0], list):
            first = 1
            for i, filters in enumerate(self.filters):
                for j, filter_ in enumerate(filters):
                    if first:
                        # Create the first convolution layer
                        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=filter_, kernel_size=[3, 3],
                                                stride=1, padding=1, dilation=1, groups=1, bias=True))
                        layers.append(nn.BatchNorm2d(num_features=filter_, eps=1e-05))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout2d(p=0.2))
                        first = 0

                    else:
                        if j == 0 and i != 0:
                            layers.append(nn.Conv2d(in_channels=self.filters[i-1][0], out_channels=filter_,
                                                    kernel_size=[3, 3], stride=1, padding=1, dilation=1, groups=1,
                                                    bias=True))
                            layers.append(nn.BatchNorm2d(num_features=filter_, eps=1e-05))
                            layers.append(nn.ReLU())
                            layers.append(nn.Dropout2d(p=0.2))

                        else:
                            layers.append(nn.Conv2d(in_channels=filters[j - 1], out_channels=filter_,
                                                    kernel_size=[3, 3], stride=1, padding=1, dilation=1, groups=1,
                                                    bias=True))
                            layers.append(nn.BatchNorm2d(num_features=filter_, eps=1e-05))
                            layers.append(nn.ReLU())
                            layers.append(nn.Dropout2d(p=0.2))

                if (i + 1) % self.pool_every == 0:
                    layers.append(nn.MaxPool2d(kernel_size=[2, 2], stride=None, padding=0, dilation=1))

        else:
            for i, filters in enumerate(self.filters):
                if i == 0:
                    # Create the first convolution layer
                    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=[3, 3],
                                            stride=1, padding=1, dilation=1, groups=1, bias=True))
                    layers.append(nn.BatchNorm2d(num_features=filters, eps=1e-05))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout2d(p=0.2))

                else:
                    layers.append(nn.Conv2d(in_channels=self.filters[i - 1], out_channels=filters, kernel_size=[3, 3],
                                            stride=1, padding=1, dilation=1, groups=1, bias=True))
                    layers.append(nn.BatchNorm2d(num_features=filters, eps=1e-05))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout2d(p=0.2))

                if (i + 1) % self.pool_every == 0:
                    layers.append(nn.MaxPool2d(kernel_size=[2, 2], stride=None, padding=0, dilation=1))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):

        layers = []
        num_of_layers = len(self.filters)
        num_of_pools = num_of_layers // self.pool_every

        if num_of_pools == 0:
            num_of_in_channels = int((self.in_size[1]) ** 2)

        else:
            num_of_in_channels = int((self.in_size[1] // (2 * num_of_pools)) ** 2)

        for i, dims in enumerate(self.hidden_dims):
            if i == 0:
                if isinstance(self.filters[-1], list):
                    f = self.filters[-1][0]

                else:
                    f = self.filters[-1]

                layers.append(nn.Linear(in_features=(f * num_of_in_channels), out_features=dims,
                                        bias=True))
                layers.append(nn.ReLU())

            else:
                layers.append(nn.Linear(in_features=self.hidden_dims[i - 1], out_features=dims, bias=True))
                layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=self.hidden_dims[-1], out_features=self.out_classes, bias=True))

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        out = self.feature_extractor.forward(x)
        out = torch.reshape(out, (out.size()[0], -1))
        out = self.classifier.forward(out)
        # ========================
        
        return out
    # ========================


