#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""A More Flexible Video models."""

import torch
import torch.nn as nn

import slowfast.utils.weight_init_helper as init_helper

from .build import MODEL_REGISTRY
from .custom_helper import ResNetBackbone, Bottleneck

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}


@MODEL_REGISTRY.register()
class CNNCatLSTM(nn.Module):
    """
    CNNCatLSTM model builder for CNN concatenate with LSTM.
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(CNNCatLSTM, self).__init__()
        # self.norm_module = get_norm(cfg)
        # self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a CNNCatLSTM model. CNN model is used for extract spatial features
            and LSTM model process temporal information.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]
        num_classes = cfg.MODEL.NUM_CLASSES
        dropout_rate = cfg.MODEL.DROPOUT_RATE
        act_func = cfg.MODEL.HEAD_ACT

        self.backbone = ResNetBackbone(Bottleneck, (d2, d3, d4, d5))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 512),
        )
        self.lstm = nn.LSTM(512, 256)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(256, num_classes, bias=True)
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x, bboxes=None):
        x = x[0]
        # batch, c, t, h, w = [x.item() for x in x.shape]
        batch, c, t, h, w = x.shape
        x = x.permute([0, 2, 1, 3, 4])
        x = x.contiguous()
        x = x.view((batch * t, c, h, w))
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(batch, t, -1)

        x = self.embedding(x)

        x, (hn, cn) = self.lstm(x)
        x = x[:, -1, :]

        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)

        return x
