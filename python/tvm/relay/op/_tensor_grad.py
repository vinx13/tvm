# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import
from ..expr import const, Tuple, TupleGetItem, TupleWrapper
from .op import register_gradient
from .reduce import sum
from .transform import collapse_sum_like, broadcast_to_like, reshape_like, transpose, where, take, reshape, tile
from . import nn as _nn
from .nn import dense, avg_pool2d_grad
from .tensor import exp, negative, power, less, log, shape_of
from .tensor import zeros_like, ones_like
from . import nn as _nn
import topi
from topi.util import get_const_tuple
from topi.nn.util import get_pad_tuple


@register_gradient("log")
def log_grad(orig, grad):
    """Returns [grad * (1 / x)]"""
    x = orig.args[0]
    return [grad * ones_like(x) / x]

@register_gradient("cos")
def cos_grad(orig, grad):
    """Returns [grad * (-sin(x))]"""
    x = orig.args[0]
    ones = ones_like(x)
    return [grad * (-ones * sin(x))]

@register_gradient("sin")
def sin_grad(orig, grad):
    """Returns [grad * cos(x)]"""
    x = orig.args[0]
    return [grad * cos(x)]

@register_gradient("exp")
def exp_grad(orig, grad):
    """Returns [grad * exp(x)]"""
    return [grad * exp(orig.args[0])]


@register_gradient("sqrt")
def sqrt_grad(orig, grad):
    """Returns [grad * 0.5 * (x ^ -0.5)]"""
    a = const(0.5)  # (TODO) type?
    return [grad * a * power(orig.args[0], negative(a))]


@register_gradient("sigmoid")
def sigmoid_grad(orig, grad):
    """Returns [grad * sigmoid(x) * (1 - sigmoid(x))]."""
    return [grad * orig * (ones_like(orig) - orig)]


@register_gradient("tanh")
def tanh_grad(orig, grad):
    """Returns grad * (1 - tanh(x) * tanh(x))."""
    return [grad * ones_like(orig) - orig * orig]


@register_gradient("nn.relu")
def relu_grad(orig, grad):
    """Returns grad * (select(x < 0, 0, 1))."""
    x = orig.args[0]
    zeros = zeros_like(x)
    ones = ones_like(x)
    return [where(less(x, zeros), zeros, ones * grad)]


@register_gradient("add")
def add_grad(orig, grad):
    """Returns [grad, grad]"""
    return [collapse_sum_like(grad, orig.args[0]),
            collapse_sum_like(grad, orig.args[1])]


@register_gradient("subtract")
def subtract_grad(orig, grad):
    """Returns [grad, -grad]"""
    return [collapse_sum_like(grad, orig.args[0]),
            collapse_sum_like(negative(grad), orig.args[1])]


@register_gradient("multiply")
def multiply_grad(orig, grad):
    """Returns [grad * y, grad * x]"""
    x, y = orig.args
    return [collapse_sum_like(grad * y, x),
            collapse_sum_like(grad * x, y)]


@register_gradient("divide")
def divide_grad(orig, grad):
    """Returns [grad / y,  - grad * (x / y) / y]"""
    x, y = orig.args
    return [collapse_sum_like(grad / y, x),
            collapse_sum_like(- (grad * orig / y), y)]


@register_gradient("zeros")
def zeros_grad(orig, grad):
    """Returns []"""
    return []


@register_gradient("ones")
def ones_grad(orig, grad):
    """Returns []"""
    return []


@register_gradient("zeros_like")
def zeros_like_grad(orig, grad):
    """Returns [0]"""
    return [orig]


@register_gradient("ones_like")
def ones_like_grad(orig, grad):
    """Returns [0]"""
    return [zeros_like(orig.args[0])]


@register_gradient("collapse_sum_like")
def collapse_sum_like_grad(orig, grad):
    """Returns [broadcast_to_like(grad, x), 0]"""
    x, y = orig.args
    return [broadcast_to_like(grad, x), zeros_like(y)]


@register_gradient("abs")
def abs_grad(orig, grad):
    """Returns grad * (select(x < 0, -1, 1))."""
    x = orig.args[0]
    zeros = zeros_like(x)
    ones = ones_like(x)
    return [where(less(x, zeros), -ones * grad, ones * grad)]


@register_gradient("clip")
def clip_grad(orig, grad):
    """Returns grad * (select(x < min || max < x , 0, 1))."""
    x = orig.args[0]
    a_min = orig.attrs.get_int("a_min")
    a_max = orig.attrs.get_int("a_max")
    a_mins = broadcast_to_like(const(a_min), x)
    a_maxs = broadcast_to_like(const(a_max), x)
    zeros = zeros_like(x)
    ones = ones_like(x)
    return [where(less(x, a_mins), zeros, where(less(a_maxs, x), zeros, ones * grad))]


@register_gradient("nn.max_pool2d")
def max_pool2d_grad(orig, grad):
    attrs = orig.attrs
    pool_grad = _nn.max_pool2d_grad(grad, orig.args[0], pool_size=attrs.pool_size,
                                    strides=attrs.strides, padding=attrs.padding,
                                    layout=attrs.layout, ceil_mode=attrs.ceil_mode)
    return [pool_grad]


@register_gradient("nn.avg_pool2d")
def avg_pool2d_grad(orig, grad):
    attrs = orig.attrs
    pool_grad = _nn.avg_pool2d_grad(grad, orig.args[0], pool_size=attrs.pool_size,
                                    strides=attrs.strides, padding=attrs.padding,
                                    layout=attrs.layout, ceil_mode=attrs.ceil_mode,
                                    count_include_pad=attrs.count_include_pad)
    return [pool_grad]

# not implemented, this is only for testing.
@register_gradient("concatenate")
def concatenate_grad(orig, grad):
    assert len(orig.args) == 1
    t = orig.args[0]
    x = TupleGetItem(t, 0)
    y = TupleGetItem(t, 1)
    # Assume only two element in tuple rn.
    # In the real implementation, concatenate_grad probably need to be implemented by an operator.
    return [Tuple([zeros_like(x), zeros_like(y)])]

# WIP by @vinx13
# confident
@register_gradient("nn.dropout")
def dropout_grad(orig, grad):
    mask = TupleWrapper(orig, 2)[1]
    return [grad * mask]

@register_gradient("nn.batch_flatten")
def batch_flatten_grad(orig, grad):
    data = orig.args[0]
    return [reshape_like(grad, data)]

@register_gradient("expand_dims")
def expand_dims_grad(orig, grad):
    data = orig.args[0]
    return [reshape_like(grad, data)]

@register_gradient("squeeze")
def squeeze_grad(orig, grad):
    data = orig.args[0]
    return [reshape_like(grad, data)]

@register_gradient("nn.global_avg_pool2d")
def global_avg_pool2d_gradient(orig, grad):
    return [ones_like(orig.args[0]) * grad]


# untested
@register_gradient("nn.dense")
def dense_grad(orig, grad):
    data, weight = orig.args
    return [collapse_sum_like(dense(grad, transpose(weight)), data),
            collapse_sum_like(transpose(dense(transpose(data), transpose(grad))), weight)]


@register_gradient("nn.conv2d")
def conv2d_grad(orig, grad):
    data_orig = orig.args[0]
    out_grad = grad

    attrs = orig.attrs
    data, weight = orig.args
    data_shape = get_const_tuple(data.checked_type.shape)
    weight_shape = get_const_tuple(weight.checked_type.shape)
    _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(get_const_tuple(attrs.padding), (filter_h, filter_w))
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    backward_data = _nn.conv2d_transpose(grad, weight, 
                                         strides=attrs.strides, 
                                         padding=attrs.padding, 
                                         dilation=attrs.dilation, 
                                         groups=attrs.groups, 
                                         data_layout=attrs.data_layout, 
                                         kernel_layout=attrs.kernel_layout, 
                                         output_padding=output_padding,
                                         out_dtype=attrs.out_dtype)

    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])
    data = reshape(data, [1, -1, 0, 0])
    backward_weight = _nn.conv2d(data, grad,
                                 strides=attrs.dilation,
                                 padding=attrs.padding,
                                 dilation=attrs.strides,
                                 groups=in_channel * batch)
    backward_weight = reshape(backward_weight, [batch, in_channel, out_channel, filter_h, filter_w]) # FIXME: using filter_h/w is not correct here if strides are not divisible
    backward_weight = sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    # TOPI-based version
    #backward_data = _nn.conv2d_backward_data(weight, out_grad, data_shape, strides=attrs.strides, padding=attrs.padding, dilation=attrs.dilation)
    #backward_weight = _nn.conv2d_backward_weight(data_orig, out_grad, weight_shape, strides=attrs.strides, padding=attrs.padding, dilation=attrs.dilation)
    return [backward_data, backward_weight]

@register_gradient("nn.cross_entropy")
def cross_entropy_grad(orig, grad):
    x, y = orig.args
    sm = _nn.softmax(x)
    batch_size = x.checked_type.shape[0].value
    grad = grad / const(batch_size, dtype='float32')
    return [(sm - y), log(sm)]
