import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from lib import fa_linear
from lib import linear
import os

BATCH_SIZE = 32

# load mnist dataset
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ])),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ])),
                         batch_size=BATCH_SIZE, shuffle=True)

# load feedforward dfa model
model_fa = fa_linear.LinearFANetwork(in_features=784, num_layers=2, num_hidden_list=[1000, 10])

# load reference linear model
model_bp = linear.LinearNetwork(in_features=784, num_layers=2, num_hidden_list=[1000, 10])

# optimizers
optimizer_fa = torch.optim.SGD(model_fa.parameters(),
                            lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True)
optimizer_bp = torch.optim.SGD(model_bp.parameters(),
                            lr=1e-4, momentum=0.9, weight_decay=0.001, nesterov=True)

loss_crossentropy = torch.nn.CrossEntropyLoss()

# make log file
results_path = 'bp_vs_fa_'
logger_train = open(results_path + 'train_log.txt', 'w')

# train loop
epochs = 1000
for epoch in xrange(epochs):
    for idx_batch, (inputs, targets) in enumerate(train_loader):
        # flatten the inputs from square image to 1d vector
        inputs = inputs.view(BATCH_SIZE, -1)
        # wrap them into varaibles
        inputs, targets = Variable(inputs), Variable(targets)
        # get outputs from the model
        outputs_fa = model_fa(inputs)
        outputs_bp = model_bp(inputs)
        # calculate loss
        loss_fa = loss_crossentropy(outputs_fa, targets)
        loss_bp = loss_crossentropy(outputs_bp, targets)

        model_fa.zero_grad()
        loss_fa.backward()
        optimizer_fa.step()

        model_bp.zero_grad()
        loss_bp.backward()
        optimizer_bp.step()

        if (idx_batch + 1) % 10 == 0:
            train_log = 'epoch ' + str(epoch) + ' step ' + str(idx_batch + 1) + \
                        ' loss_fa ' + str(loss_fa.data[0]) + ' loss_bp ' + str(loss_bp.data[0])
            print(train_log)
            logger_train.write(train_log + '\n')
