import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable


class LinearFANetwork(nn.Module):
    """
    Linear feed-forward networks with feedback alignment learning
    Does NOT perform non-linear activation after each layer
    """
    def __init__(self, in_features, num_layers, num_hidden_list):
        """
        :param in_features: dimension of input features (784 for MNIST)
        :param num_layers: number of layers for feed-forward net
        :param num_hidden_list: list of integers indicating hidden nodes of each layer
        """
        super(LinearFANetwork, self).__init__()
        self.in_features = in_features
        self.num_layers = num_layers
        self.num_hidden_list = num_hidden_list

        # create list of linear layers
        # first hidden layer
        self.linear = [LinearFAModule(self.in_features, self.num_hidden_list[0])]
        # append additional hidden layers to list
        for idx in xrange(self.num_layers - 1):
            self.linear.append(LinearFAModule(self.num_hidden_list[idx], self.num_hidden_list[idx+1]))

        # create ModuleList to make list of layers work
        self.linear = nn.ModuleList(self.linear)

    def forward(self, inputs):
        """
        forward pass, which is same for conventional feed-forward net
        :param inputs: inputs with shape [batch_size, in_features]
        :return: logit outputs from the network
        """

        # first layer
        linear1 = self.linear[0](inputs)

        # second layer
        linear2 = self.linear[1](linear1)

        return linear2


class LinearFAFunction(autograd.Function):

    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_variables
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            # all of the logic of FA resides in this one line
            # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
            grad_input = grad_output.mm(weight_fa)
        if context.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias


class LinearFAModule(nn.Module):

    def __init__(self, input_features, output_features, bias=True):
        super(LinearFAModule, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # weight and bias for forward pass
        # weight has transposed form; more efficient (so i heard) (transposed at forward pass)
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # fixed random weight and bias for FA backward pass
        # does not need gradient
        self.weight_fa = Variable(torch.FloatTensor(output_features, input_features), requires_grad=False)

        # weight initialization
        torch.nn.init.kaiming_uniform(self.weight)
        torch.nn.init.kaiming_uniform(self.weight_fa)
        torch.nn.init.constant(self.bias, 1)

    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)




