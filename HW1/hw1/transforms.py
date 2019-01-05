import torch
import numpy as np

class TensorView(object):
    """
    A transform that returns a new view of a tensor.
    """

    def __init__(self, *view_dims):
        self.view_dims = view_dims

    def __call__(self, tensor: torch.Tensor):
        # TODO: Use Tensor.view() to implement the transform.
        # ====== YOUR CODE: ======
        # First validate the the dimensions are valid
        current_dims = list(tensor.shape)
        num_of_elements = np.prod(np.array(current_dims))
        
        zeros_inds = np.where(self.view_dims == 0)
        zeros_inds = zeros_inds[0]
        
        if zeros_inds.size:
            raise ValueError("Cannot specify an axis dimension as '0', all axis dimension must be either real positive integers, or '-1'")
        
        neg_inds = np.where(np.array(self.view_dims) < 0)
        neg_inds = neg_inds[0]
        
        if neg_inds.size:
            pos_dims = [dim for dim in self.view_dims if dim > 0]
            specified_elements = np.prod(np.array(pos_dims))
            
            mod = np.mod(num_of_elements, specified_elements)
            
            if mod != 0:
                raise ValueError("Axis dimension chosen are incompatible with the original dimension of a Tensor with shape: {}".format(self.view_dims))
                
        # If we've reached here then everything is ok with the choosen dimensions
        return tensor.view(self.view_dims)
        # ========================


class BiasTrick(object):
    """
    A transform that applies the "bias trick": Adds an element equal to 1 to
    a given tensor.
    """

    def __call__(self, tensor: torch.Tensor):
        assert tensor.dim() == 1, "Only 1-d tensors supported"

        # TODO: Add a 1 at the end of the given tensor.
        # Make sure to use the same data type.

        # ====== YOUR CODE: ======
        return torch.cat([tensor, torch.Tensor(np.array([1,]), ).type_as(tensor)])
        # ========================


