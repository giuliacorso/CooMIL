# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
import torch.optim as optim
import sys
from utils.magic import persistent_locals
import wandb

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        # self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.opt = optim.Adam(self.net.parameters(), lr=self.args.lr)
        self.device = get_device()

    def forward(self, x) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass


class ContinualModelCoCoopMil(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModelCoCoopMil, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = get_device()
        exclude_from_optimizer = set(args.exclude_from_optimizer)
        exclude_from_optimizer2 = set(args.exclude_from_optimizer2)
        params = []
        learnable_names=[]
        for name, param in self.net.named_parameters():
            names= set(name.split('.'))
            if len(names.intersection(exclude_from_optimizer))>0:
                continue
            else:
                params.append(param)
                learnable_names.append(name)
        if self.args.coomil_optimizer == 'adam':
            self.opt = optim.Adam(params, lr=args.coomil_lr)
        elif self.args.coomil_optimizer == 'sgd':
            self.opt = optim.SGD(params, lr=args.coomil_lr)

        params = []
        learnable_names=[]
        for name, param in self.net.named_parameters():
            names = set(name.split('.'))
            if len(names.intersection(exclude_from_optimizer2)) > 0:
                continue
            else:
                params.append(param)
                learnable_names.append(name)
        #self.opt2= optim.SGD(params, lr=0.1)


    def forward(self, x) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :return: the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        #if 'wandb' in sys.modules and not self.args.nowand:
        #    pl = persistent_locals(self.observe)
        #    ret = pl(*args, **kwargs)
        #    self.autolog_wandb(pl.locals)
        #else:
        #    ret = self.observe(*args, **kwargs)
        ret = self.observe(*args, **kwargs)
        return ret

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor, t: int, ssl: bool) -> float:
        """
        Compute a training step over a given batch of examples.
        inputs: batch of examples
        labels: ground-truth labels
        kwargs: some methods could require additional parameters
        t: task index
        ssl: True if the model is trained with ssl
        :return: the value of the loss function
        """
        raise NotImplementedError

    def autolog_wandb(self, locals):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            wandb.log({k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
                       for k, v in locals.items() if k.startswith('_wandb_') or k.startswith('loss')})

    def save_model(self):
        pass