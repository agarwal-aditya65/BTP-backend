"""``unet_256`` generator, identical to junyanz/pytorch-CycleGAN-and-pix2pix.

Your three pix2pix runs (rain / low-light / fog) all used ``--netG unet_256
--ngf 64 --input_nc 3 --output_nc 3`` and dropout enabled (``no_dropout False``).
They differ only in ``--norm`` (batch for rain + low-light, instance for fog),
which changes the layer biases and the saved keys – so the norm type MUST match
the one used when each model was trained.

Layer names/nesting (``model.model.1.model...``) are kept exactly as in the
original repo so the ``*_net_G.pth`` files load with ``strict=True``.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn


def get_norm_layer(norm_type: str = "batch"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super().__init__()
        self.outermost = outermost
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downconv] + [submodule] + [uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv] + [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, downnorm] + [submodule] + [uprelu, upconv, upnorm]
            if use_dropout:
                model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):
            block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=block, norm_layer=norm_layer, use_dropout=use_dropout)
        block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=block,
                                             outermost=True, norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


def build_generator(norm: str = "batch") -> UnetGenerator:
    """``unet_256`` generator exactly as built by the original repo's ``define_G``."""
    return UnetGenerator(3, 3, 8, 64, norm_layer=get_norm_layer(norm), use_dropout=True)
