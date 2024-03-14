"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

from abc import abstractmethod

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod # 抽象方法表示基类的一个方法，没有实现，所以基类不能实例化，子类实现了该抽象方法才能被实例化。
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
