import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.utils.checkpoint import checkpoint
from .hrfpn import HRFPN

from ..builder import NECKS


@NECKS.register_module()
class HRFPNplus(HRFPN):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 num_extra_outs=1,
                 pooling_type='AVG',
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False,
                 stride=1,):
        super(HRFPNplus, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            pooling_type,
            conv_cfg,
            norm_cfg,
            with_cp,
            stride)
        self.num_extra_outs = num_extra_outs
        self.extra_fpn_convs = nn.ModuleList()
        for i in range(self.num_extra_outs):
            self.extra_fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    act_cfg=None))

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        outs = [inputs[0]]
        for i in range(1, self.num_ins):
            outs.append(
                F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
        if out.requires_grad and self.with_cp:
            out = checkpoint(self.reduction_conv, out)
        else:
            out = self.reduction_conv(out)

        extra_outs = []
        for i in range(self.num_extra_outs):
            extra_outs.append(F.interpolate(
                out,
                scale_factor=2**(self.num_extra_outs - i),
                mode='bilinear'))

        outs = [out]
        for i in range(1, self.num_outs):
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))

        outputs = []

        for i in range(self.num_extra_outs):
            if extra_outs[i].requires_grad and self.with_cp:
                tmp_out = checkpoint(self.extra_fpn_convs[i], extra_outs[i])
            else:
                tmp_out = self.extra_fpn_convs[i](extra_outs[i])
            outputs.append(tmp_out)

        for i in range(self.num_outs):
            if outs[i].requires_grad and self.with_cp:
                tmp_out = checkpoint(self.fpn_convs[i], outs[i])
            else:
                tmp_out = self.fpn_convs[i](outs[i])
            outputs.append(tmp_out)
        return tuple(outputs)
