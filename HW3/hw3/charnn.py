import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO: Create two maps as described in the docstring above.
    # It's best if you also sort the chars before assigning indices, so that
    # they're in lexical order.
    # ====== YOUR CODE: ======
    # Sort the text based on ASCII values
    sorted_text = sorted(text)

    # Get only the unique chars in the text
    unique_chars = []
    for x in sorted_text:
        if x not in unique_chars:
            unique_chars.append(x)

    # Create the two dictionaries
    char_to_idx = {}
    idx_to_char = {}
    for i in range(len(unique_chars)):
        char_to_idx[unique_chars[i]] = i
        idx_to_char[i] = unique_chars[i]

    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    # Filter out all of the unwanted chars and return as a list
    text_clean = list(filter(lambda ch: ch not in chars_to_remove, text))

    # Turn the list into a string
    text_clean = ''.join(text_clean)

    # Calculate how many chars were removed
    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    # Calculate the number of total & unique chars
    D = len(char_to_idx)
    N = len(text)
    result = torch.zeros((N, D), dtype=torch.int8, requires_grad=False)

    # Fill in the encoding
    for i in range(N):
        result[i, char_to_idx[text[i]]] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    # Calculate the number of total & unique chars
    N = embedded_text.shape[0]
    result = ''

    # Fill in the encoding
    idxs = torch.argmax(input=embedded_text, dim=1, keepdim=False)

    for i in range(N):
        result += idx_to_char[int(idxs[i])]
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO: Implement the labelled samples creation.
    # 1. Embed the given text.
    # 2. Create the samples tensor by splitting to groups of seq_len.
    #    Notice that the last char has no label, so don't use it.
    # 3. Create the labels tensor in a similar way and convert to indices.
    # Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    # Embed the text
    embedded_text = chars_to_onehot(text=text, char_to_idx=char_to_idx)

    # Remove the encoding for the last char since it doesn't have an appropriate label
    samples = embedded_text[0:-1, :]

    # Create the labels embedding vector
    labels = torch.argmax(input=embedded_text, dim=1)
    labels = labels[1:]

    # Create the samples and labels tensors
    labels = labels.unfold(dimension=0, size=seq_len, step=seq_len)
    labels = labels.to(device)
    samples = samples.unfold(dimension=0, size=seq_len, step=seq_len)

    # Re-Order the axis in order to fit the required shape
    samples = samples.permute(0, 2, 1)
    samples = samples.to(device)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO: Implement char-by-char text generation.
    # 1. Feed the start_sequence into the model.
    # 2. Sample a new char from the output distribution of the last output
    #    char. Convert output to probabilities first.
    #    See torch.multinomial() for the sampling part.
    # 3. Feed the new char into the model.
    # 4. Rinse and Repeat.
    #
    # Note that tracking tensor operations for gradient calculation is not
    # necessary for this. Best to disable tracking for speed.
    # See torch.no_grad().
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================

    return out_text


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of output dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model.
        # To implement the affine transforms you can use either nn.Linear
        # modules (recommended) or create W and b tensor pairs directly.
        # Create these modules or tensors and save them per-layer in
        # the layer_params list.
        # Important note: You must register the created parameters so
        # they are returned from our module's parameters() function.
        # Usually this happens automatically when we assign a
        # module/tensor as an attribute in our module, but now we need
        # to do it manually since we're not assigning attributes. So:
        #   - If you use nn.Linear modules, call self.add_module() on them
        #     to register each of their parameters as part of your model.
        #   - If you use tensors directly, wrap them in nn.Parameter() and
        #     then call self.register_parameter() on them. Also make
        #     sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        # Create the hidden states vector for the input - i.e. h_0 & initialize it with the He (Kaiming) normal
        # initialization.
        h_0 = torch.zeros(in_dim, h_dim)
        h_0 = nn.init.kaiming_normal_(h_0, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        # Register the input hidden layer into the model
        self.register_parameter(name="h_0", param=nn.Parameter(h_0))

        for i in range(n_layers):
            # Create all of the weights matrices for each layer, for all of the GRU Cell units & initialize all of them
            # with the He (Kaiming) normal initialization.
            W_xz_i = torch.zeros(out_dim, in_dim)
            W_xz_i = nn.init.kaiming_normal_(W_xz_i, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            W_hz_i = torch.zeros(out_dim, h_dim)
            W_hz_i = nn.init.kaiming_normal_(W_hz_i, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            W_xr_i = torch.zeros(out_dim, in_dim)
            W_xr_i = nn.init.kaiming_normal_(W_xr_i, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            W_hr_i = torch.zeros(out_dim, h_dim)
            W_hr_i = nn.init.kaiming_normal_(W_hr_i, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            W_xg_i = torch.zeros(out_dim, in_dim)
            W_xg_i = nn.init.kaiming_normal_(W_xg_i, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            W_hg_i = torch.zeros(out_dim, h_dim)
            W_hg_i = nn.init.kaiming_normal_(W_hg_i, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

            # Create all of the hidden states vectors for each layer & initialize all of them with the He (Kaiming)
            # normal initialization.
            h_i = torch.zeros(in_dim, h_dim)
            h_i = nn.init.kaiming_normal_(h_i, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

            # Create all of the biases vectors for each layer, for all of the GRU Cell units & initialize all of them
            # with the He (Kaiming) normal initialization.
            B_z_i = torch.zeros(out_dim)
            B_z_i = nn.init.kaiming_normal_(B_z_i, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            B_r_i = torch.zeros(out_dim)
            B_r_i = nn.init.kaiming_normal_(B_r_i, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            B_g_i = torch.zeros(out_dim)
            B_g_i = nn.init.kaiming_normal_(B_g_i, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

            # Register all of the parameters to the model
            self.register_parameter(name=f"W_xz_{i + 1}", param=nn.Parameter(W_xz_i))
            self.register_parameter(name=f"W_hz_{i + 1}", param=nn.Parameter(W_hz_i))
            self.register_parameter(name=f"W_xr_{i + 1}", param=nn.Parameter(W_xr_i))
            self.register_parameter(name=f"W_hr_{i + 1}", param=nn.Parameter(W_hr_i))
            self.register_parameter(name=f"W_xg_{i + 1}", param=nn.Parameter(W_xg_i))
            self.register_parameter(name=f"W_hg_{i + 1}", param=nn.Parameter(W_hg_i))
            self.register_parameter(name=f"h_{i + 1}", param=nn.Parameter(h_i))
            self.register_parameter(name=f"B_z_{i + 1}", param=nn.Parameter(B_z_i))
            self.register_parameter(name=f"B_r_{i + 1}", param=nn.Parameter(B_r_i))
            self.register_parameter(name=f"B_g_{i + 1}", param=nn.Parameter(B_g_i))

        # Create the weights matrix and bias vector for the output layer & initialize it with the He (Kaiming) normal
        # initialization.
        W_hy = torch.zeros(out_dim, in_dim)
        W_hy = nn.init.kaiming_normal_(W_hy, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        B_y = torch.zeros(out_dim)
        B_y = nn.init.kaiming_normal_(B_y, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        # Register the last layer into the model
        self.register_parameter(name="W_hy", param=nn.Parameter(W_hy))
        self.register_parameter(name="B_y", param=nn.Parameter(B_y))
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor=None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(torch.zeros(batch_size, self.h_dim, device=input.device))
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO: Implement the model's forward pass.
        # You'll need to go layer-by-layer from bottom to top (see diagram).
        # Tip: You can use torch.stack() to combine multiple tensors into a
        # single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================
        return layer_output, hidden_state
