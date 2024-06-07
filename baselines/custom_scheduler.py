import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpPiecewiseConstantSchedule(_LRScheduler):
    '''
    This class implements a learning rate scheduler that combines a warm-up phase with a piecewise constant decay.
    That is, the learning rate is increased linearly from 0 to the base learning rate during the warm-up phase, and then 
    decreased by a factor of decay_ratio at each epoch in decay_epochs (similarly to MultiStepLR).
    Note: can maybe be done better, but needed it done quickly... 
    '''
    def __init__(self, optimizer, steps_per_epoch, base_lr, lr_decay_ratio, lr_decay_epochs, warmup_epochs, last_epoch=-1):
        self.steps_per_epoch = steps_per_epoch
        self.base_lr = base_lr
        self.decay_ratio = lr_decay_ratio
        self.decay_epochs = lr_decay_epochs
        self.warmup_epochs = warmup_epochs
        self.decay_steps = [e * steps_per_epoch for e in lr_decay_epochs]  # Convert epochs to steps
        super(WarmUpPiecewiseConstantSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Calculate the current step
        lr_step = self.last_epoch
        lr_epoch = lr_step / self.steps_per_epoch
        learning_rate = self.base_lr

        # Warm-Up Phase
        if lr_epoch < self.warmup_epochs:
            learning_rate = self.base_lr * lr_step / (self.warmup_epochs * self.steps_per_epoch)
        else:
            # Piecewise Constant Decay Phase
            for i, start_step in enumerate(self.decay_steps):
                if lr_step >= start_step:
                    learning_rate = self.base_lr * (self.decay_ratio ** (i + 1))
                else:
                    break

        return [learning_rate for _ in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr