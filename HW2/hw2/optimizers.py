import abc
import torch
from torch import Tensor


class Optimizer(abc.ABC):
    """
    Base class for optimizers.
    """
    def __init__(self, params):
        """
        :param params: A sequence of model parameters to optimize. Can be a
        list of (param,grad) tuples as returned by the Blocks, or a list of
        pytorch tensors in which case the grad will be taken from them.
        """
        assert isinstance(params, list) or isinstance(params, tuple)
        self._params = params

    @property
    def params(self):
        """
        :return: A sequence of parameter tuples, each tuple containing
        (param_data, param_grad). The data should be updated in-place
        according to the grad.
        """
        returned_params = []
        for x in self._params:
            if isinstance(x, Tensor):
                p = x.data
                dp = x.grad.data if x.grad is not None else None
                returned_params.append((p, dp))
            elif isinstance(x, tuple) and len(x) == 2:
                returned_params.append(x)
            else:
                raise TypeError(f"Unexpected parameter type for parameter {x}")

        return returned_params

    def zero_grad(self):
        """
        Sets the gradient of the optimized parameters to zero (in place).
        """
        for p, dp in self.params:
            dp.zero_()

    @abc.abstractmethod
    def step(self):
        """
        Updates all the registered parameter values based on their gradients.
        """
        raise NotImplementedError()


class VanillaSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue
            
            # TODO: Implement the optimizer step.
            # Update the gradient according to regularization and then
            # update the parameters tensor.
            # ====== YOUR CODE: ======
            # The parameters' gradients using the regularization is:
            # dp' = dp + lambda * p
            dp += (self.reg * p)
            
            # Update the parameter according to the gradient descent rule:
            # theta = theta - learning_rate * gradient_of_theta
            p -= (self.learn_rate * dp)         
            # ========================


class MomentumSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, momentum=0.9):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param momentum: Momentum factor
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.momentum = momentum

        # TODO: Add your own initializations as needed.
        # ====== YOUR CODE: ======
        # Create an indicator for which parameters we are currently working with
        self.block_id = 0
        
        # Initialize the momentum at t = 0 to be v_0 = 0
        self.v = []
        for p, dp in self.params:
            if dp is None:
                continue
            
            self.v.append(torch.zeros_like(dp))
        # ========================

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # update the parameters tensor based on the velocity. Don't forget
            # to include the regularization term.
            # ====== YOUR CODE: ======
            # The parameters' gradients using the regularization is:
            dp += (self.reg * p)
            
            # Compute v_(t+1) & update it for the next iteration
            v = self.momentum * self.v[self.block_id] - self.learn_rate * dp
            self.v[self.block_id] = v
            
            # Update the parameter according to the gradient descent with momentum rule:
            # theta = theta + v_(t + 1)
            p += v     
            
            # Advance our counter
            self.block_id += 1
        
        # Don't forget to return our blocks iterator to 0 
        self.block_id = 0
            # ========================


class RMSProp(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, decay=0.99, eps=1e-8):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param decay: Gradient exponential decay factor
        :param eps: Constant to add to gradient sum for numerical stability
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.decay = decay
        self.eps = eps

        # TODO: Add your own initializations as needed.
        # ====== YOUR CODE: ======
        # Create an indicator for which parameters we are currently working with
        self.block_id = 0
        
        # Initialize the RMS momentum at t = 0 to be r_0 = 0
        self.r = []
        for p, dp in self.params:
            if dp is None:
                continue
            
            self.r.append(torch.zeros_like(dp))
        # ========================

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # Create a per-parameter learning rate based on a decaying moving
            # average of it's previous gradients. Use it to update the
            # parameters tensor.
            # ====== YOUR CODE: ======
            # The parameters' gradients using the regularization is:
            dp += (self.reg * p)
            
            # Compute r_(t+1) & update it for the next iteration
            r = self.decay * self.r[self.block_id] + (1 - self.decay) * torch.pow(input=dp, exponent=2)
            self.r[self.block_id] = r
            
            # Update the parameter according to the RMS rule:
            # theta = theta - (laerning_rate / sqrt(r_(t + 1) + eps)) * grad_theta
            sqrt = torch.sqrt(input=(self.r[self.block_id] + self.eps))
            p -= dp * (self.learn_rate * torch.pow(input=sqrt, exponent=-1))  
            
            # Advance our counter
            self.block_id += 1
        
        # Don't forget to return our blocks iterator to 0 
        self.block_id = 0
            # ========================
