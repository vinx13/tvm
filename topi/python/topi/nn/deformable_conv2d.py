# pylint: disable=invalid-name, too-many-locals, too-many-arguments
"""Deformable Conv2D operators"""
import tvm

from .pad import pad
from .util import get_pad_tuple
from ..util import get_const_tuple

@tvm.target.generic_func
def deformable_conv2d_nchw(data, offset, kernel, strides, padding, dilation, deformable_groups,
                           out_dtype):
    """Deformable conv2D operator in NCHW layout.

    The deformable convolution operation is described in https://arxiv.org/abs/1703.06211

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    offset : tvm.Tensor
        4-D with shape [batch, deformable_groups * filter_height * filter_width * 2,
                        out_height, out_width]

    kernel : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    deformable_groups : int
        number of deformable groups

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = data.dtype

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = get_const_tuple(data.shape)
    out_channel, channel, kernel_h, kernel_w = get_const_tuple(kernel.shape)
    _, _, out_height, out_width = get_const_tuple(offset.shape)
    assert in_channel % deformable_groups == 0, "Input cahnnels must divide deformable group size"
    ic_per_dgroup = channel // deformable_groups

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, _, _ = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')

    zero = tvm.const(0.0, data.dtype)

    def _bilinear(n, c, h, w):
        non_zero = tvm.all(h >= 0, w >= 0, h < in_height, w < in_width)

        low_h = h.astype('int32')
        low_w = w.astype('int32')
        high_h = tvm.min(low_h + 1, in_height - 1)
        high_w = tvm.min(low_w + 1, in_height - 1)
        y_lerp = h - low_h
        x_lerp = w - low_w
        interpolate = (1 - y_lerp) * (1 - x_lerp) * data(n, c, low_h, low_w) + \
                   (1 - y_lerp) * x_lerp * data(n, c, low_h, high_w) + \
                   y_lerp * (1 - x_lerp) * data(n, c, high_h, low_w) + \
                   y_lerp * x_lerp * data(n, c, high_h, high_w)

        return tvm.select(non_zero, interpolate, zero)

    data_deform = \
        tvm.compute((batch, out_channel, out_height, out_width, kernel_h, kernel_w),
                    lambda n, c, y, x, kh, kw:
                    _bilinear(n, c,
                              y * stride_h - pad_top + kh * dilation_h +
                              offset[n, c // ic_per_dgroup * kernel_h * kernel_w * 2 +
                                     (kh * kernel_w + kw) * 2, y, x],
                              x * stride_w - pad_left + kw * dilation_w +
                              offset[n, c // ic_per_dgroup * kernel_h * kernel_w * 2 +
                                     (kh * kernel_w + kw) * 2 + 1, y, x]))

    return tvm.compute(
        (batch, out_channel, out_height, out_width),
        lambda n, f, y, x: tvm.sum(
            data_deform[n, rc, y, x, ry, rx].astype(out_dtype) *
            kernel[f, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag="deformable_conv2d_nchw")
