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
# pylint: disable=invalid-name,unused-argument
"""Schedule template of deformable conv2d with cuda backend"""
import tvm
from tvm import te
from tvm import autotvm
from .. import nn
from ..nn.util import get_pad_tuple
from ..util import traverse_inline, get_const_tuple
from ..cpp.util import bilinear_sample_nchw, bilinear_sample_nhwc

from tvm.contrib import nvcc
import os
import numpy as np

TASK="gemm"
USE_MANUAL_CODE = True

@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code):
    write_code(code, '/home/ubuntu/ws/code.cu')
    ptx =  nvcc.compile_cuda(code, target="ptx")
    return ptx

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

def intrin_wmma_load_matrix(scope):
    n = 16
    A = te.placeholder((n, n), name='A', dtype='float16')
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope='shared', data_alignment=32, offset_factor=256)
    C = te.compute((n, n), lambda i, j: A[i, j], name='C')
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(tvm.tir.call_intrin('handle', 'tvm_load_matrix_sync',
                                BC.data, n, n, n, BC.elem_offset // 256,
                                BA.access_ptr('r'), n, 'row_major'))
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def _intrin_wmma_gemm():
    n = 16
    A = te.placeholder((n, n), name='A', dtype='float16')
    B = te.placeholder((n, n), name='B', dtype='float16')
    k = te.reduce_axis((0, n), name="k")
    C = te.compute((n, n),
                    lambda ii, jj:
                    te.sum(A[ii, k].astype('float') * B[k, jj].astype('float'), axis=k),
                    name='C')
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, name='BA', scope='wmma.matrix_a', data_alignment=32, offset_factor=256)
    BB = tvm.tir.decl_buffer(B.shape, B.dtype, name='BB', scope='wmma.matrix_b', data_alignment=32, offset_factor=256)
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, name='BC', scope='wmma.accumulator', data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        BA, BB = ins
        BC, = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_intrin('handle', 'tvm_fill_fragment', BC.data, n, n, n, BC.elem_offset // 256, 0.0))
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_intrin('handle', 'tvm_mma_sync',
                                    BC.data, BC.elem_offset // 256,
                                    BA.data, BA.elem_offset // 256,
                                    BB.data, BB.elem_offset // 256,
                                    BC.data, BC.elem_offset // 256))
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def intrin_wmma_store_matrix():
    n = 16
    A = te.placeholder((n, n), name='A', dtype='float32')
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope='wmma.accumulator', data_alignment=32, offset_factor=256)
    C = te.compute((n, n), lambda i, j: A[i, j], name='C')
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope='global', data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(tvm.tir.call_intrin('handle', 'tvm_store_matrix_sync',
                                BA.data, n, n, n, BA.elem_offset // 256,
                                BC.access_ptr('w'), n, 'row_major'))
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})
#@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code

#NHWC = True

def deformable_conv2d_half_tensorcore(data, offset, kernel, strides, padding, dilation, deformable_groups,
                                      groups, out_dtype):
    """Deformable conv2D operator in NCHW layout.

    The deformable convolution operation is described in https://arxiv.org/abs/1703.06211

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    offset : tvm.te.Tensor
        4-D with shape [batch, deformable_groups * filter_height * filter_width * 2,
        out_height, out_width].

    kernel : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    deformable_groups : int
        number of deformable groups

    groups : int
        number of groups

    Returns
    -------
    output : tvm.te.Tensor
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

    block_size = 16
    batch, in_channel, in_height, in_width = get_const_tuple(data.shape)
    kernel_h, kernel_w, channel, out_channel_div_block_size, _, _ = get_const_tuple(kernel.shape) # out_channel is actually out_channel // 16
    out_channel = out_channel_div_block_size * block_size
    _, _, out_height, out_width = get_const_tuple(offset.shape)
    assert in_channel % deformable_groups == 0, "Input cahnnels must divide deformable group size"
    assert groups == 1, "deformable_conv2d_nchw does not support groups > 1"

    ic_per_dgroup = channel * block_size // deformable_groups

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, _, _ = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    rco = te.reduce_axis((0, in_channel//block_size), name='rco')
    rci = te.reduce_axis((0, block_size), name='rci')
    ry = te.reduce_axis((0, kernel_h), name='ry')
    rx = te.reduce_axis((0, kernel_w), name='rx')

    zero = tvm.tir.const(0.0, data.dtype)

    def _bilinear(n, c, h, w):
        outside = tvm.tir.any(h < 0, w < 0, h >= in_height, w >= in_width)
        val = bilinear_sample_nchw(data, (n, c, h, w), in_height - 1, in_width - 1)
        return tvm.tir.if_then_else(outside, zero, val)

    data_deform = \
        te.compute((batch, in_channel, kernel_h, kernel_w, out_height, out_width),
                   lambda n, c, kh, kw, y, x:
                   _bilinear(n, c,
                             y * stride_h - pad_top + kh * dilation_h +
                             offset[n, c // ic_per_dgroup * (kernel_w*kernel_h*2)+
                                    (kh * kernel_w + kw) * 2, y, x],
                             x * stride_w - pad_left + kw * dilation_w +
                             offset[n, c // ic_per_dgroup * (kernel_w*kernel_h*2) +
                                    (kh * kernel_w + kw) * 2 + 1, y, x]), tag="data_deform", name='data_deform')
    data_deform_packed = te.compute((batch, in_channel // block_size, kernel_h, kernel_w, out_height, out_width, block_size), 
    lambda n,co,kh,kw,h,w, ci: data_deform[n, co * block_size + ci, kh, kw, h, w], tag='data_deform_packed', name='data_deform_packed')
    # return packed output
    return te.compute((batch, out_channel, out_height, out_width), lambda n, f, h, w:
    te.sum(data_deform_packed[n, rco, ry, rx, h, w, rci].astype('float32') * 
    kernel[ry, rx, rco, tvm.tir.indexdiv(f, block_size), rci, tvm.tir.indexmod(f, block_size)].astype('float32'), axis=[rco,ry,rx,rci]), tag='deformable_conv2d_nchw')
    # return te.compute(
    #     (batch, out_channel, out_height, out_width),
    #     lambda n, f, y, x: te.sum(
    #         data_deform[n, rc, ry, rx, y, x].astype(out_dtype) *
    #         kernel[f, rc, ry, rx].astype(out_dtype),
    #         axis=[rc, ry, rx]), tag="deformable_conv2d_nchw")

def _schedule_tensorcore(s, Conv):
    data_deform_packed, W = s[Conv].op.input_tensors
    ico, kh, kw, ici = s[Conv].op.reduce_axis
    data_deform = s[data_deform_packed].op.input_tensors[0]
    in_dtype = 'float16'
    out_dtype = 'float32'
    s[data_deform].compute_inline()

    AS = s.cache_read(data_deform_packed, 'shared', [Conv])
    s[data_deform_packed].compute_inline()
    AF = s.cache_read(AS, 'wmma.matrix_a', [Conv])
    WS = s.cache_read(W, 'shared', [Conv])
    WF = s.cache_read(WS, 'wmma.matrix_b', [Conv])
    ConvF = s.cache_write(Conv, 'wmma.accumulator')
    
    if Conv.op in s.outputs:
        output = Conv
        ConvS = s.cache_read(ConvF, 'shared', [Conv])
        OL = ConvS
    else:
        output = s.outputs[0].output(0)
        s[Conv].set_scope('shared')
        OL = Conv

    block_row_warps = 4
    block_col_warps = 2
    warp_row_tiles = 2
    warp_col_tiles = 2
    #block_row_warps = 1
    #block_col_warps = 1
    #warp_row_tiles = 2
    #warp_col_tiles = 2
    warp_size = 32
    chunk = 2
    wmma_m = wmma_n = wmma_k = 16
    offset = 0
    vector_width = 2

    block_x = te.thread_axis('blockIdx.x')
    block_y = te.thread_axis('blockIdx.y')
    block_z = te.thread_axis('blockIdx.z')
    thread_x = te.thread_axis('threadIdx.x')
    thread_y = te.thread_axis('threadIdx.y')
    thread_z = te.thread_axis('threadIdx.z')

    # Define the intrin strides
    def get_strides(extents):
        return [np.prod(extents[i:]).tolist() for i in range(len(extents))]

    AS_align = chunk * wmma_k + offset
    WS_align = warp_col_tiles * block_col_warps * wmma_n + offset
    block_factor_w = warp_row_tiles * block_row_warps # also removed wmma_m here because we switched to h instead
    block_factor_o = warp_col_tiles * block_col_warps * wmma_n # wmma_n is implied in shape
    CS_align = block_factor_o + offset
    AS_strides = get_strides([1, 1, AS_align, 1])
    AL_strides = get_strides([1, 1, wmma_k, 1])
    #WS_strides = get_strides([WS_align, 1])
    #WL_strides = get_strides([wmma_n * warp_col_tiles, 1])
    # my ASAL strides
    AS_strides = get_strides([wmma_k, 1])
    AL_strides = get_strides([wmma_k, 1])
    # my WSWL strides
    WS_strides = get_strides([wmma_k, 1])
    WL_strides = get_strides([wmma_k, 1])
    # end 
    CL_strides = get_strides([warp_row_tiles, wmma_n, 1])
    CS_strides = get_strides([block_row_warps*warp_row_tiles, wmma_n, 1])

    #wci = wi
    #kernel_scope, n = s[conv].split(n, nparts=1)

    # Schedule for output
    nc, oc, hc, wc = s[Conv].op.axis
    block_i, hc = s[output].split(hc, factor=block_factor_w)
    wco, wci = s[output].split(wc, factor=wmma_m)
    block_j, oco = s[output].split(oc, factor=block_factor_o)
    s[output].reorder(nc, wco, block_i, block_j, hc, wci, oco)
    block_k = s[output].fuse(nc, wco)
    s[output].bind(block_k, block_z)
    s[output].reorder(block_k, block_i, block_j, hc, oco)

    t = s[output].fuse(hc, wci, oco) # FIXME: order?
    #t, ti = s[output].split(t, factor=2)
    t, tx = s[output].split(t, factor=warp_size)
    t, ty = s[output].split(t, factor=block_row_warps)
    t, tz = s[output].split(t, factor=block_col_warps) 
    s[output].bind(block_i, block_x)
    s[output].bind(block_j, block_y)
    s[output].bind(tz, thread_z)
    s[output].bind(ty, thread_y)
    s[output].bind(tx, thread_x)
    #s[output].vectorize(ti)

    # Schedule wmma store
    s[OL].compute_at(s[output], block_j)
    nc, oc, hc, wc = OL.op.axis
    #s[OL].reorder(nc, hc, wc, oci) # FIXME
    #s[OL].storage_align(wc, CS_align - 1, CS_align)
    oc, ooc = s[OL].split(oc, factor=wmma_n)
    oco, oci = s[OL].split(oc, factor=warp_col_tiles)
    _, oco = s[OL].split(oco, factor=block_col_warps)
    wc, wwc = s[OL].split(wc, factor=wmma_m)
    hc, hci = s[OL].split(hc, factor=warp_row_tiles)
    _, hc = s[OL].split(hc, factor=block_row_warps)
    s[OL].reorder(oco, hc, hci, oci, wc, wwc, ooc)
    s[OL].bind(hc, thread_y)
    s[OL].bind(oco, thread_z)

    # Schedule wmma computation
    s[ConvF].compute_at(s[OL], oco)
    n, o, h, w = ConvF.op.axis
    o, oof = s[ConvF].split(o, factor=wmma_n)
    w, wwf = s[ConvF].split(w, factor=wmma_m)
    #ic, ii = s[ConvF].split(ico, factor=wmma_k)
    ko, ki = s[ConvF].split(ico, factor=chunk)
    s[ConvF].reorder(kh, kw, ko, ki, n, o, h, w, wwf, oof, ici)

    s[AF].compute_at(s[ConvF], ki)
    s[WF].compute_at(s[ConvF], ki)

    # Schedule wmma load
    n, i, rh, rw, h, w, ii = AF.op.axis
    w, ww = s[AF].split(w, factor=wmma_m)
    s[AF].reorder(w, ww, ii)

    kh, kw, i, o, ii, oo = WF.op.axis
    # i, ii = s[WF].split(i, factor=wmma_k)
    # o, oo = s[WF].split(o, factor=wmma_n)
    # s[WF].reorder(o, i, oo)
    # s[WF].reorder(i, o, ii, oo)

    s[WS].compute_at(s[ConvF], ko)
    s[AS].compute_at(s[ConvF], ko)

    # Schedule for data's share memory
    n, rco, rh, rw, h, w, rci = AS.op.axis
    #s[AS].reorder(h, w, n, i)
    #s[AS].storage_align(w, AS_align - 1, AS_align)
    # t = s[AS].fuse(n, i)
    t = s[AS].fuse(*s[AS].op.axis)
    #t, ti = s[AS].split(t, factor=vector_width)
    t, tx = s[AS].split(t, factor=warp_size)
    t, ty = s[AS].split(t, factor=block_row_warps)
    _, tz = s[AS].split(t, factor=block_col_warps)
    s[AS].bind(ty, thread_y)
    s[AS].bind(tz, thread_z)
    s[AS].bind(tx, thread_x)
    #s[AS].vectorize(ti)

    # Schedule for kernel's share memory
    kh, kw, ic, oc, _, _ = WS.op.axis
    # t = s[WS].fuse(ic, oc)
    # s[WS].storage_align(ic, WS_align - 1, WS_align)
    t = s[WS].fuse(*s[WS].op.axis)
    t, ti = s[WS].split(t, factor=vector_width)
    t, tx = s[WS].split(t, factor=warp_size)
    t, ty = s[WS].split(t, factor=block_row_warps)
    _, tz = s[WS].split(t, factor=block_col_warps)
    s[WS].bind(ty, thread_y)
    s[WS].bind(tz, thread_z)
    s[WS].bind(tx, thread_x)
    s[WS].vectorize(ti)

    shape = (wmma_m, wmma_n, wmma_k)

    # tensorize the wmma process
    AS_shape = (wmma_m, wmma_k)
    AL_shape = (wmma_m, wmma_k)
    WS_shape = (wmma_k, wmma_n)
    WL_shape = (wmma_k, wmma_n)
    CL_shape = (wmma_m, 1, wmma_n)
    CS_shape = (wmma_m, 1, wmma_n)

    AL_gemm = te.placeholder(AL_shape, name='A', dtype=in_dtype)
    WL_gemm = te.placeholder(WL_shape, name='B', dtype=in_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k")
    CL_compute = te.compute(CL_shape, lambda ii, yy, jj:
                            te.sum(AL_gemm[jj, k_gemm].astype(out_dtype) * \
                                   WL_gemm[k_gemm, ii].astype(out_dtype), axis=k_gemm),
                            name='C')

    s[AF].tensorize(ww, intrin_wmma_load_matrix_A(AL_strides, AS_strides, shape,
                                                  "row_major", AS_shape, AL_shape, in_dtype))
    s[WF].tensorize(ii, intrin_wmma_load_matrix_W(WL_strides, WS_strides, shape,
                                                  "row_major", WS_shape, WL_shape, in_dtype))
    s[OL].tensorize(wwc, intrin_wmma_store_matrix(CS_strides, CL_strides,
                                                  shape, 'col_major', out_dtype, CL_shape, CS_shape))
    s[ConvF].tensorize(wwf, intrin_wmma_gemm(AL_gemm, WL_gemm, CL_compute, AL_strides,
                                             WL_strides, CL_strides, shape))



def schedule_deformable_conv2d_half_tensorcore(outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'deformable_conv2d_nchw':
            #_schedule_direct_cuda(cfg, s, op.output(0))
            _schedule_tensorcore(s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def deformable_conv2d_nchw_cuda(data, offset, kernel, strides, padding, dilation, deformable_groups,
                                groups, out_dtype):
    """Deformable conv2D operator in NCHW layout.

    The deformable convolution operation is described in https://arxiv.org/abs/1703.06211

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    offset : tvm.te.Tensor
        4-D with shape [batch, deformable_groups * filter_height * filter_width * 2,
        out_height, out_width].

    kernel : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    deformable_groups : int
        number of deformable groups

    groups : int
        number of groups

    Returns
    -------
    output : tvm.te.Tensor
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
    assert groups == 1, "deformable_conv2d_nchw does not support groups > 1"

    ic_per_dgroup = channel // deformable_groups

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, _, _ = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    rc = te.reduce_axis((0, in_channel), name='rc')
    ry = te.reduce_axis((0, kernel_h), name='ry')
    rx = te.reduce_axis((0, kernel_w), name='rx')

    zero = tvm.tir.const(0.0, data.dtype)

    def _bilinear(n, c, h, w):
        outside = tvm.tir.any(h < 0, w < 0, h >= in_height, w >= in_width)
        val = bilinear_sample_nchw(data, (n, c, h, w), in_height - 1, in_width - 1)
        return tvm.tir.if_then_else(outside, zero, val)

    data_deform = \
        te.compute((batch, in_channel, kernel_h, kernel_w, out_height, out_width),
                   lambda n, c, kh, kw, y, x:
                   _bilinear(n, c,
                             y * stride_h - pad_top + kh * dilation_h +
                             offset[n, c // ic_per_dgroup * (kernel_w*kernel_h*2) +
                                    (kh * kernel_w + kw) * 2, y, x],
                             x * stride_w - pad_left + kw * dilation_w +
                             offset[n, c // ic_per_dgroup * (kernel_w*kernel_h*2) +
                                    (kh * kernel_w + kw) * 2 + 1, y, x]), tag="data_deform", name='data_deform')
    return te.compute(
        (batch, out_channel, out_height, out_width),
        lambda n, f, y, x: te.sum(
            data_deform[n, rc, ry, rx, y, x].astype(out_dtype) *
            kernel[f, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag="deformable_conv2d_nchw")

def deformable_conv2d_deformed_input(data_deform, kernel, strides, padding, dilation, deformable_groups, groups, out_dtype):
    batch, in_channel, kernel_h, kernel_w, out_height, out_width = get_const_tuple(data_deform.shape)
    out_channel = get_const_tuple(kernel.shape)[0]
    rc = te.reduce_axis((0, in_channel), name='rc')
    ry = te.reduce_axis((0, kernel_h), name='ry')
    rx = te.reduce_axis((0, kernel_w), name='rx')
    return te.compute(
        (batch, out_channel, out_height, out_width),
        lambda n, f, y, x: te.sum(
            data_deform[n, rc, ry, rx, y, x].astype(out_dtype) *
            kernel[f, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag="deformable_conv2d_nchw")

def _schedule_deformable_conv2d_deformed_input(s, conv):
    data_deform, kernel = s[conv].op.input_tensors
    if conv.op in s.outputs:
        # output is conv
        conv_local = s.cache_write(conv, 'local')
    else:
        s[conv].set_scope('local')
        conv_local = conv
        # output is relu but named conv
        conv = s.outputs[0].output(0)

    n, f, y, x = tuple(conv.op.axis)
    n_c, f_c, y_c, x_c, rc, ry, rx = tuple(conv_local.op.axis) + tuple(conv_local.op.reduce_axis)
    n_c_o_i, n_c_i = s[conv_local].split(n_c, factor=1)
    n_c_o_o_i, n_c_o_i = s[conv_local].split(n_c_o_i, factor=1)
    n_c_o_o_o_i, n_c_o_o_i = s[conv_local].split(n_c_o_o_i, factor=1)
    n_c_o_o_o_o, n_c_o_o_o_i = s[conv_local].split(n_c_o_o_o_i, factor=1)
    f_c_o_i, f_c_i = s[conv_local].split(f_c, factor=4)
    f_c_o_o_i, f_c_o_i = s[conv_local].split(f_c_o_i, factor=1)
    f_c_o_o_o_i, f_c_o_o_i = s[conv_local].split(f_c_o_o_i, factor=32)
    f_c_o_o_o_o, f_c_o_o_o_i = s[conv_local].split(f_c_o_o_o_i, factor=1)
    y_c_o_i, y_c_i = s[conv_local].split(y_c, factor=2)
    y_c_o_o_i, y_c_o_i = s[conv_local].split(y_c_o_i, factor=1)
    y_c_o_o_o_i, y_c_o_o_i = s[conv_local].split(y_c_o_o_i, factor=1)
    y_c_o_o_o_o, y_c_o_o_o_i = s[conv_local].split(y_c_o_o_o_i, factor=1)
    x_c_o_i, x_c_i = s[conv_local].split(x_c, factor=4)
    x_c_o_o_i, x_c_o_i = s[conv_local].split(x_c_o_i, factor=1)
    x_c_o_o_o_i, x_c_o_o_i = s[conv_local].split(x_c_o_o_i, factor=8)
    x_c_o_o_o_o, x_c_o_o_o_i = s[conv_local].split(x_c_o_o_o_i, factor=1)
    rc_o_i, rc_i = s[conv_local].split(rc, factor=2)
    rc_o_o, rc_o_i = s[conv_local].split(rc_o_i, factor=2)
    ry_o_i, ry_i = s[conv_local].split(ry, factor=3)
    ry_o_o, ry_o_i = s[conv_local].split(ry_o_i, factor=1)
    rx_o_i, rx_i = s[conv_local].split(rx, factor=3)
    rx_o_o, rx_o_i = s[conv_local].split(rx_o_i, factor=1)
    s[conv_local].reorder(n_c_o_o_o_o, f_c_o_o_o_o, y_c_o_o_o_o, x_c_o_o_o_o, n_c_o_o_o_i, f_c_o_o_o_i, y_c_o_o_o_i, x_c_o_o_o_i, n_c_o_o_i, f_c_o_o_i, y_c_o_o_i, x_c_o_o_i, rc_o_o, ry_o_o, rx_o_o, rc_o_i, ry_o_i, rx_o_i, n_c_o_i, f_c_o_i, y_c_o_i, x_c_o_i, rc_i, ry_i, rx_i, n_c_i, f_c_i, y_c_i, x_c_i)
    #s[conv_local].reorder(rc_o_o, ry_o_o, rx_o_o, rc_o_i, ry_o_i, rx_o_i, n_c, f_c, y_c, x_c,rc_i, ry_i, rx_i)
    n_o_i, n_i = s[conv].split(n, factor=1)
    n_o_o_i, n_o_i = s[conv].split(n_o_i, factor=1)
    n_o_o_o, n_o_o_i = s[conv].split(n_o_o_i, factor=1)
    f_o_i, f_i = s[conv].split(f, factor=4)
    f_o_o_i, f_o_i = s[conv].split(f_o_i, factor=32)
    f_o_o_o, f_o_o_i = s[conv].split(f_o_o_i, factor=1)
    y_o_i, y_i = s[conv].split(y, factor=2)
    y_o_o_i, y_o_i = s[conv].split(y_o_i, factor=1)
    y_o_o_o, y_o_o_i = s[conv].split(y_o_o_i, factor=1)
    x_o_i, x_i = s[conv].split(x, factor=4)
    x_o_o_i, x_o_i = s[conv].split(x_o_i, factor=8)
    x_o_o_o, x_o_o_i = s[conv].split(x_o_o_i, factor=1)
    s[conv].reorder(n_o_o_o, f_o_o_o, y_o_o_o, x_o_o_o, n_o_o_i, f_o_o_i, y_o_o_i, x_o_o_i, n_o_i, f_o_i, y_o_i, x_o_i, n_i, f_i, y_i, x_i)
    n_c_o_o_o_o_f_c_o_o_o_o_fused_y_c_o_o_o_o_fused_x_c_o_o_o_o_fused = s[conv_local].fuse(n_c_o_o_o_o, f_c_o_o_o_o, y_c_o_o_o_o, x_c_o_o_o_o)
    n_o_o_o_f_o_o_o_fused_y_o_o_o_fused_x_o_o_o_fused = s[conv].fuse(n_o_o_o, f_o_o_o, y_o_o_o, x_o_o_o)
    n_c_o_o_o_i_f_c_o_o_o_i_fused_y_c_o_o_o_i_fused_x_c_o_o_o_i_fused = s[conv_local].fuse(n_c_o_o_o_i, f_c_o_o_o_i, y_c_o_o_o_i, x_c_o_o_o_i)
    n_o_o_i_f_o_o_i_fused_y_o_o_i_fused_x_o_o_i_fused = s[conv].fuse(n_o_o_i, f_o_o_i, y_o_o_i, x_o_o_i)
    n_c_o_o_i_f_c_o_o_i_fused_y_c_o_o_i_fused_x_c_o_o_i_fused = s[conv_local].fuse(n_c_o_o_i, f_c_o_o_i, y_c_o_o_i, x_c_o_o_i)
    n_o_i_f_o_i_fused_y_o_i_fused_x_o_i_fused = s[conv].fuse(n_o_i, f_o_i, y_o_i, x_o_i)
    s[conv_local].compute_at(s[conv], n_o_i_f_o_i_fused_y_o_i_fused_x_o_i_fused)
    kernel_shared = s.cache_read(kernel, "shared", [conv_local])
    ax0, ax1, ax2, ax3 = tuple(kernel_shared.op.axis)
    s[kernel_shared].compute_at(s[conv_local], rx_o_o)
    ax0_ax1_fused_ax2_fused_ax3_fused = s[kernel_shared].fuse(ax0, ax1, ax2, ax3)
    ax0_ax1_fused_ax2_fused_ax3_fused_o, ax0_ax1_fused_ax2_fused_ax3_fused_i = s[kernel_shared].split(ax0_ax1_fused_ax2_fused_ax3_fused, factor=256)
    s[kernel_shared].bind(ax0_ax1_fused_ax2_fused_ax3_fused_i, te.thread_axis("threadIdx.x"))
    data_deform_shared = s.cache_read(data_deform, "shared", [conv_local])
    ax0, ax1, ax2, ax3, ax4, ax5 = tuple(data_deform_shared.op.axis)
    s[data_deform_shared].compute_at(s[conv_local], rx_o_o)

    # we want to check the loops first, bind n
    #s[data_deform_shared].reorder(ax0, ax2, ax3, ax4, ax5, ax1)
    #ax0_o, ax0_i = s[data_deform_shared].split(ax0, factor=256)
    #s[data_deform_shared].bind(ax0_i, te.thread_axis('threadIdx.x'))
    #s[data_deform_shared].vectorize(ax1)
    
    #ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused = s[data_deform_shared].fuse(ax0, ax2, ax3, ax4, ax5, ax1)
    ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused = s[data_deform_shared].fuse(ax0, ax1, ax2, ax3, ax4, ax5)

    #ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused, ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_v = s[data_deform_shared].split(ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused, factor=4)
    #s[data_deform_shared].vectorize(ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_v)
    ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_o, ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_i = s[data_deform_shared].split(ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused, factor=256)
    s[data_deform_shared].bind(ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_i, te.thread_axis("threadIdx.x"))


    s[conv].bind(n_o_o_o_f_o_o_o_fused_y_o_o_o_fused_x_o_o_o_fused, te.thread_axis("blockIdx.x"))
    #s[conv].bind(n_o_o_i_f_o_o_i_fused_y_o_o_i_fused_x_o_o_i_fused, te.thread_axis("vthread")) # it is useless because vthread=1
    s[conv].bind(n_o_i_f_o_i_fused_y_o_i_fused_x_o_i_fused, te.thread_axis("threadIdx.x"))
    s[conv_local].pragma(n_c_o_o_o_o_f_c_o_o_o_o_fused_y_c_o_o_o_o_fused_x_c_o_o_o_o_fused, "auto_unroll_max_step", 512)
    s[conv_local].pragma(n_c_o_o_o_o_f_c_o_o_o_o_fused_y_c_o_o_o_o_fused_x_c_o_o_o_o_fused, "unroll_explicit", True)
    # return s

def schedule_deformable_conv2d_deformed_input(outs):
    """TOPI schedule callback of deformable conv2d for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'deformable_conv2d_nchw':
            #_schedule_direct_cuda(cfg, s, op.output(0))
            print('MANUAL')
            _schedule_deformable_conv2d_deformed_input(s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s
    
def deformable_conv2d_nhwc_nchw_nchw_cuda(data, offset, kernel, strides, padding, dilation, deformable_groups,
                                          groups, out_dtype):
    """Deformable conv2D operator in NHWC data, NCHW offset, NCHW output layout.

    The deformable convolution operation is described in https://arxiv.org/abs/1703.06211

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    offset : tvm.te.Tensor
        4-D with shape [batch, deformable_groups * filter_height * filter_width * 2,
        out_height, out_width].

    kernel : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    deformable_groups : int
        number of deformable groups

    groups : int
        number of groups

    Returns
    -------
    output : tvm.te.Tensor
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

    batch, in_height, in_width, in_channel = get_const_tuple(data.shape)
    out_channel, channel, kernel_h, kernel_w = get_const_tuple(kernel.shape)
    _, _, out_height, out_width = get_const_tuple(offset.shape)
    assert in_channel % deformable_groups == 0, "Input cahnnels must divide deformable group size"
    assert groups == 1, "deformable_conv2d_nchw does not support groups > 1"

    ic_per_dgroup = channel // deformable_groups

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, _, _ = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    rc = te.reduce_axis((0, in_channel), name='rc')
    ry = te.reduce_axis((0, kernel_h), name='ry')
    rx = te.reduce_axis((0, kernel_w), name='rx')

    zero = tvm.tir.const(0.0, data.dtype)

    def _bilinear(n, c, h, w):
        outside = tvm.tir.any(h < 0, w < 0, h >= in_height, w >= in_width)
        val = bilinear_sample_nhwc(data, (n, h, w, c), in_height - 1, in_width - 1)
        return tvm.tir.if_then_else(outside, zero, val)

    data_deform = \
        te.compute((batch, in_channel, kernel_h, kernel_w, out_height, out_width),
                   lambda n, c, kh, kw, y, x:
                   _bilinear(n, c,
                             y * stride_h - pad_top + kh * dilation_h +
                             offset[n, c // ic_per_dgroup * (kernel_w*kernel_h*2) +
                                    (kh * kernel_w + kw) * 2, y, x],
                             x * stride_w - pad_left + kw * dilation_w +
                             offset[n, c // ic_per_dgroup * (kernel_w*kernel_h*2) +
                                    (kh * kernel_w + kw) * 2 + 1, y, x]), tag="data_deform", name='data_deform')
    return te.compute(
        (batch, out_channel, out_height, out_width),
        lambda n, f, y, x: te.sum(
            data_deform[n, rc, ry, rx, y, x].astype(out_dtype) *
            kernel[f, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag="deformable_conv2d_nhwc_nchw_nchw")

@autotvm.register_topi_compute("deformable_conv2d_nchw.cuda")
def deformable_conv2d_nchw(cfg, data, offset, kernel, strides, padding, dilation,
                           deformable_groups, groups, out_dtype):
    return deformable_conv2d_nchw_cuda(data, offset, kernel, strides, padding, dilation,
                                     deformable_groups, groups, out_dtype)

@autotvm.register_topi_compute("deformable_conv2d_nhwc_nchw_nchw.cuda")
def deformable_conv2d_nhwc_nchw_nchw(cfg, data, offset, kernel, strides, padding, dilation,
                                     deformable_groups, groups, out_dtype):
    return deformable_conv2d_nhwc_nchw_nchw_cuda(data, offset, kernel, strides, padding, dilation,
                                                 deformable_groups, groups, out_dtype)

@autotvm.register_topi_schedule("deformable_conv2d_nhwc_nchw_nchw.cuda")
def schedule_deformable_conv2d_nhwc_nchw_nchw(cfg, outs):
    """TOPI schedule callback of deformable conv2d for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'deformable_conv2d_nhwc_nchw_nchw':
            _schedule_nhwc_cuda(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s

@autotvm.register_topi_schedule("deformable_conv2d_nchw.cuda")
def schedule_deformable_conv2d_nchw(cfg, outs):
    """TOPI schedule callback of deformable conv2d for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'deformable_conv2d_nchw':
            #_schedule_direct_cuda(cfg, s, op.output(0))
            print('MANUAL')
            _schedule_manual_cuda(s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_nhwc_cuda(cfg, s, conv):
    """Schedule template of deformable conv2d"""
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.Target.current()
    if target.target_name in ['nvptx', 'rocm']:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])

    data_deform, kernel = s[conv].op.input_tensors

    s[data_deform].compute_inline()
    if isinstance(kernel.op, tvm.te.ComputeOp) and 'dilate' in kernel.op.tag:
        s[kernel].compute_inline()

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, 'local')
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope('local')
        OL = conv

    # create cache stage
    AA = s.cache_read(data_deform, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg['tile_rc'].apply(s, OL, rc)
    ryo, ryi = cfg['tile_ry'].apply(s, OL, ry)
    rxo, rxi = cfg['tile_rx'].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x)
    cfg.define_reorder("reorder_inner", [rco, ryo, rxo], "all")
    cfg["reorder_inner"].apply(s, OL, [rco, ryo, rxo])
    cfg["reorder_inner"].apply(s, OL, [rci, ryi, rxi])

    cache_loc = [rco, ryo, rxo][cfg["reorder_inner"].perm[-1]]
    s[AA].compute_at(s[OL], cache_loc)
    s[WW].compute_at(s[OL], cache_loc)

    # cooperative fetching
    for load in [AA, WW]:
        fused = s[load].fuse(*s[load].op.axis)
        if load == AA:
            fused, fused_v = s[load].split(fused, factor=4)
            s[load].vectorize(fused_v)

        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))


    # unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)


def _schedule_manual_cuda(s, conv):
    data_deform, kernel = s[conv].op.input_tensors
    print(s[data_deform].op.input_tensors)
    if conv.op in s.outputs:
        # output is conv
        conv_local = s.cache_write(conv, 'local')
    else:
        s[conv].set_scope('local')
        conv_local = conv
        # output is relu but named conv
        conv = s.outputs[0].output(0)

    n, c, kh, kw, _, _ = tuple(data_deform.op.axis)
    n, f, y, x = tuple(conv.op.axis)
    n_c, f_c, y_c, x_c, rc, ry, rx = tuple(conv_local.op.axis) + tuple(conv_local.op.reduce_axis)
    n_c_o_i, n_c_i = s[conv_local].split(n_c, factor=1)
    n_c_o_o_i, n_c_o_i = s[conv_local].split(n_c_o_i, factor=1)
    n_c_o_o_o_i, n_c_o_o_i = s[conv_local].split(n_c_o_o_i, factor=1)
    n_c_o_o_o_o, n_c_o_o_o_i = s[conv_local].split(n_c_o_o_o_i, factor=1)
    f_c_o_i, f_c_i = s[conv_local].split(f_c, factor=4)
    f_c_o_o_i, f_c_o_i = s[conv_local].split(f_c_o_i, factor=1)
    f_c_o_o_o_i, f_c_o_o_i = s[conv_local].split(f_c_o_o_i, factor=32)
    f_c_o_o_o_o, f_c_o_o_o_i = s[conv_local].split(f_c_o_o_o_i, factor=1)
    y_c_o_i, y_c_i = s[conv_local].split(y_c, factor=2)
    y_c_o_o_i, y_c_o_i = s[conv_local].split(y_c_o_i, factor=1)
    y_c_o_o_o_i, y_c_o_o_i = s[conv_local].split(y_c_o_o_i, factor=1)
    y_c_o_o_o_o, y_c_o_o_o_i = s[conv_local].split(y_c_o_o_o_i, factor=1)
    x_c_o_i, x_c_i = s[conv_local].split(x_c, factor=4)
    x_c_o_o_i, x_c_o_i = s[conv_local].split(x_c_o_i, factor=1)
    x_c_o_o_o_i, x_c_o_o_i = s[conv_local].split(x_c_o_o_i, factor=8)
    x_c_o_o_o_o, x_c_o_o_o_i = s[conv_local].split(x_c_o_o_o_i, factor=1)
    rc_o_i, rc_i = s[conv_local].split(rc, factor=2)
    rc_o_o, rc_o_i = s[conv_local].split(rc_o_i, factor=2)
    ry_o_i, ry_i = s[conv_local].split(ry, factor=3)
    ry_o_o, ry_o_i = s[conv_local].split(ry_o_i, factor=1)
    rx_o_i, rx_i = s[conv_local].split(rx, factor=3)
    rx_o_o, rx_o_i = s[conv_local].split(rx_o_i, factor=1)
    s[conv_local].reorder(n_c_o_o_o_o, f_c_o_o_o_o, y_c_o_o_o_o, x_c_o_o_o_o, n_c_o_o_o_i, f_c_o_o_o_i, y_c_o_o_o_i, x_c_o_o_o_i, n_c_o_o_i, f_c_o_o_i, y_c_o_o_i, x_c_o_o_i, rc_o_o, ry_o_o, rx_o_o, rc_o_i, ry_o_i, rx_o_i, n_c_o_i, f_c_o_i, y_c_o_i, x_c_o_i, rc_i, ry_i, rx_i, n_c_i, f_c_i, y_c_i, x_c_i)
    n_o_i, n_i = s[conv].split(n, factor=1)
    n_o_o_i, n_o_i = s[conv].split(n_o_i, factor=1)
    n_o_o_o, n_o_o_i = s[conv].split(n_o_o_i, factor=1)
    f_o_i, f_i = s[conv].split(f, factor=4)
    f_o_o_i, f_o_i = s[conv].split(f_o_i, factor=32)
    f_o_o_o, f_o_o_i = s[conv].split(f_o_o_i, factor=1)
    y_o_i, y_i = s[conv].split(y, factor=2)
    y_o_o_i, y_o_i = s[conv].split(y_o_i, factor=1)
    y_o_o_o, y_o_o_i = s[conv].split(y_o_o_i, factor=1)
    x_o_i, x_i = s[conv].split(x, factor=4)
    x_o_o_i, x_o_i = s[conv].split(x_o_i, factor=8)
    x_o_o_o, x_o_o_i = s[conv].split(x_o_o_i, factor=1)
    s[conv].reorder(n_o_o_o, f_o_o_o, y_o_o_o, x_o_o_o, n_o_o_i, f_o_o_i, y_o_o_i, x_o_o_i, n_o_i, f_o_i, y_o_i, x_o_i, n_i, f_i, y_i, x_i)
    n_c_o_o_o_o_f_c_o_o_o_o_fused_y_c_o_o_o_o_fused_x_c_o_o_o_o_fused = s[conv_local].fuse(n_c_o_o_o_o, f_c_o_o_o_o, y_c_o_o_o_o, x_c_o_o_o_o)
    n_o_o_o_f_o_o_o_fused_y_o_o_o_fused_x_o_o_o_fused = s[conv].fuse(n_o_o_o, f_o_o_o, y_o_o_o, x_o_o_o)
    n_c_o_o_o_i_f_c_o_o_o_i_fused_y_c_o_o_o_i_fused_x_c_o_o_o_i_fused = s[conv_local].fuse(n_c_o_o_o_i, f_c_o_o_o_i, y_c_o_o_o_i, x_c_o_o_o_i)
    n_o_o_i_f_o_o_i_fused_y_o_o_i_fused_x_o_o_i_fused = s[conv].fuse(n_o_o_i, f_o_o_i, y_o_o_i, x_o_o_i)
    n_c_o_o_i_f_c_o_o_i_fused_y_c_o_o_i_fused_x_c_o_o_i_fused = s[conv_local].fuse(n_c_o_o_i, f_c_o_o_i, y_c_o_o_i, x_c_o_o_i)
    n_o_i_f_o_i_fused_y_o_i_fused_x_o_i_fused = s[conv].fuse(n_o_i, f_o_i, y_o_i, x_o_i)
    s[conv_local].compute_at(s[conv], n_o_i_f_o_i_fused_y_o_i_fused_x_o_i_fused)
    x_c_i, v = s[conv_local].split(x_c_i, factor=2)
    s[conv_local].vectorize(v)
    kernel_shared = s.cache_read(kernel, "shared", [conv_local])
    ax0, ax1, ax2, ax3 = tuple(kernel_shared.op.axis)
    s[kernel_shared].compute_at(s[conv_local], rx_o_o)
    ax0_ax1_fused_ax2_fused_ax3_fused = s[kernel_shared].fuse(ax0, ax1, ax2, ax3)
    #ax0_ax1_fused_ax2_fused_ax3_fused, ax0_ax1_fused_ax2_fused_ax3_fused_v = s[kernel_shared].split(ax0_ax1_fused_ax2_fused_ax3_fused, factor=2)
    #s[kernel_shared].vectorize(ax0_ax1_fused_ax2_fused_ax3_fused_v)

    ax0_ax1_fused_ax2_fused_ax3_fused_o, ax0_ax1_fused_ax2_fused_ax3_fused_i = s[kernel_shared].split(ax0_ax1_fused_ax2_fused_ax3_fused, factor=256)
    s[kernel_shared].bind(ax0_ax1_fused_ax2_fused_ax3_fused_i, te.thread_axis("threadIdx.x"))

    data_deform_shared = s.cache_read(data_deform, "shared", [conv_local])
    ax0, ax1, ax2, ax3, ax4, ax5 = tuple(data_deform_shared.op.axis)
    s[data_deform_shared].compute_at(s[conv_local], rx_o_o)

    ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused = s[data_deform_shared].fuse(ax0, ax1, ax2, ax3, ax4, ax5)

    ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_o, ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_i = s[data_deform_shared].split(ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused, factor=256)
    s[data_deform_shared].bind(ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_i, te.thread_axis("threadIdx.x"))


    s[conv].bind(n_o_o_o_f_o_o_o_fused_y_o_o_o_fused_x_o_o_o_fused, te.thread_axis("blockIdx.x"))
    s[conv].bind(n_o_o_i_f_o_o_i_fused_y_o_o_i_fused_x_o_o_i_fused, te.thread_axis("vthread")) # it is useless because vthread=1
    s[conv].bind(n_o_i_f_o_i_fused_y_o_i_fused_x_o_i_fused, te.thread_axis("threadIdx.x"))
    # n_i_f_i_y_i_x_i_fused = s[conv].fuse(n_i, f_i, y_i, x_i)
    # _, n_i_f_i_y_i_x_i_fused_v = s[conv].split(n_i_f_i_y_i_x_i_fused, factor=2) 
    #s[conv].vectorize(n_i_f_i_y_i_x_i_fused_v)
    _, v = s[conv].split(x_i, factor=2)
    s[conv].vectorize(v)
    s[conv_local].pragma(n_c_o_o_o_o_f_c_o_o_o_o_fused_y_c_o_o_o_o_fused_x_c_o_o_o_o_fused, "auto_unroll_max_step", 512)
    s[conv_local].pragma(n_c_o_o_o_o_f_c_o_o_o_o_fused_y_c_o_o_o_o_fused_x_c_o_o_o_o_fused, "unroll_explicit", True)
    s[data_deform].compute_inline()
    # return s
    
def _schedule_manual_nhwc_cuda(s, conv):
    data_deform, kernel = s[conv].op.input_tensors
    if conv.op in s.outputs:
        # output is conv
        conv_local = s.cache_write(conv, 'local')
    else:
        s[conv].set_scope('local')
        conv_local = conv
        # output is relu but named conv
        conv = s.outputs[0].output(0)

    n, c, kh, kw, _, _ = tuple(data_deform.op.axis)
    n, f, y, x = tuple(conv.op.axis)
    n_c, f_c, y_c, x_c, rc, ry, rx = tuple(conv_local.op.axis) + tuple(conv_local.op.reduce_axis)
    n_c_o_i, n_c_i = s[conv_local].split(n_c, factor=1)
    n_c_o_o_i, n_c_o_i = s[conv_local].split(n_c_o_i, factor=1)
    n_c_o_o_o_i, n_c_o_o_i = s[conv_local].split(n_c_o_o_i, factor=1)
    n_c_o_o_o_o, n_c_o_o_o_i = s[conv_local].split(n_c_o_o_o_i, factor=1)
    f_c_o_i, f_c_i = s[conv_local].split(f_c, factor=4)
    f_c_o_o_i, f_c_o_i = s[conv_local].split(f_c_o_i, factor=1)
    f_c_o_o_o_i, f_c_o_o_i = s[conv_local].split(f_c_o_o_i, factor=32)
    f_c_o_o_o_o, f_c_o_o_o_i = s[conv_local].split(f_c_o_o_o_i, factor=1)
    y_c_o_i, y_c_i = s[conv_local].split(y_c, factor=2)
    y_c_o_o_i, y_c_o_i = s[conv_local].split(y_c_o_i, factor=1)
    y_c_o_o_o_i, y_c_o_o_i = s[conv_local].split(y_c_o_o_i, factor=1)
    y_c_o_o_o_o, y_c_o_o_o_i = s[conv_local].split(y_c_o_o_o_i, factor=1)
    x_c_o_i, x_c_i = s[conv_local].split(x_c, factor=4)
    x_c_o_o_i, x_c_o_i = s[conv_local].split(x_c_o_i, factor=1)
    x_c_o_o_o_i, x_c_o_o_i = s[conv_local].split(x_c_o_o_i, factor=8)
    x_c_o_o_o_o, x_c_o_o_o_i = s[conv_local].split(x_c_o_o_o_i, factor=1)
    rc_o_i, rc_i = s[conv_local].split(rc, factor=2)
    rc_o_o, rc_o_i = s[conv_local].split(rc_o_i, factor=2)
    ry_o_i, ry_i = s[conv_local].split(ry, factor=3)
    ry_o_o, ry_o_i = s[conv_local].split(ry_o_i, factor=1)
    rx_o_i, rx_i = s[conv_local].split(rx, factor=3)
    rx_o_o, rx_o_i = s[conv_local].split(rx_o_i, factor=1)
    s[conv_local].reorder(n_c_o_o_o_o, f_c_o_o_o_o, y_c_o_o_o_o, x_c_o_o_o_o, n_c_o_o_o_i, f_c_o_o_o_i, y_c_o_o_o_i, x_c_o_o_o_i, n_c_o_o_i, f_c_o_o_i, y_c_o_o_i, x_c_o_o_i, rc_o_o, ry_o_o, rx_o_o, rc_o_i, ry_o_i, rx_o_i, n_c_o_i, f_c_o_i, y_c_o_i, x_c_o_i, rc_i, ry_i, rx_i, n_c_i, f_c_i, y_c_i, x_c_i)
    n_o_i, n_i = s[conv].split(n, factor=1)
    n_o_o_i, n_o_i = s[conv].split(n_o_i, factor=1)
    n_o_o_o, n_o_o_i = s[conv].split(n_o_o_i, factor=1)
    f_o_i, f_i = s[conv].split(f, factor=4)
    f_o_o_i, f_o_i = s[conv].split(f_o_i, factor=32)
    f_o_o_o, f_o_o_i = s[conv].split(f_o_o_i, factor=1)
    y_o_i, y_i = s[conv].split(y, factor=2)
    y_o_o_i, y_o_i = s[conv].split(y_o_i, factor=1)
    y_o_o_o, y_o_o_i = s[conv].split(y_o_o_i, factor=1)
    x_o_i, x_i = s[conv].split(x, factor=4)
    x_o_o_i, x_o_i = s[conv].split(x_o_i, factor=8)
    x_o_o_o, x_o_o_i = s[conv].split(x_o_o_i, factor=1)
    s[conv].reorder(n_o_o_o, f_o_o_o, y_o_o_o, x_o_o_o, n_o_o_i, f_o_o_i, y_o_o_i, x_o_o_i, n_o_i, f_o_i, y_o_i, x_o_i, n_i, f_i, y_i, x_i)
    n_c_o_o_o_o_f_c_o_o_o_o_fused_y_c_o_o_o_o_fused_x_c_o_o_o_o_fused = s[conv_local].fuse(n_c_o_o_o_o, f_c_o_o_o_o, y_c_o_o_o_o, x_c_o_o_o_o)
    n_o_o_o_f_o_o_o_fused_y_o_o_o_fused_x_o_o_o_fused = s[conv].fuse(n_o_o_o, f_o_o_o, y_o_o_o, x_o_o_o)
    n_c_o_o_o_i_f_c_o_o_o_i_fused_y_c_o_o_o_i_fused_x_c_o_o_o_i_fused = s[conv_local].fuse(n_c_o_o_o_i, f_c_o_o_o_i, y_c_o_o_o_i, x_c_o_o_o_i)
    n_o_o_i_f_o_o_i_fused_y_o_o_i_fused_x_o_o_i_fused = s[conv].fuse(n_o_o_i, f_o_o_i, y_o_o_i, x_o_o_i)
    n_c_o_o_i_f_c_o_o_i_fused_y_c_o_o_i_fused_x_c_o_o_i_fused = s[conv_local].fuse(n_c_o_o_i, f_c_o_o_i, y_c_o_o_i, x_c_o_o_i)
    n_o_i_f_o_i_fused_y_o_i_fused_x_o_i_fused = s[conv].fuse(n_o_i, f_o_i, y_o_i, x_o_i)
    s[conv_local].compute_at(s[conv], n_o_i_f_o_i_fused_y_o_i_fused_x_o_i_fused)
    kernel_shared = s.cache_read(kernel, "shared", [conv_local])
    ax0, ax1, ax2, ax3 = tuple(kernel_shared.op.axis)
    s[kernel_shared].compute_at(s[conv_local], rx_o_o)
    ax0_ax1_fused_ax2_fused_ax3_fused = s[kernel_shared].fuse(ax0, ax1, ax2, ax3)
    ax0_ax1_fused_ax2_fused_ax3_fused, ax0_ax1_fused_ax2_fused_ax3_fused_v = s[kernel_shared].split(factor=2)
    s[kernel_shared].vectorize(ax0_ax1_fused_ax2_fused_ax3_fused_v)
    ax0_ax1_fused_ax2_fused_ax3_fused_o, ax0_ax1_fused_ax2_fused_ax3_fused_i = s[kernel_shared].split(ax0_ax1_fused_ax2_fused_ax3_fused, factor=256)
    s[kernel_shared].bind(ax0_ax1_fused_ax2_fused_ax3_fused_i, te.thread_axis("threadIdx.x"))
    data_deform_shared = s.cache_read(data_deform, "shared", [conv_local])
    ax0, ax1, ax2, ax3, ax4, ax5 = tuple(data_deform_shared.op.axis)
    ax5, ax5_v = s[data_deform_shared].split(ax5, factor=4)
    s[data_deform_shared].vectorize(ax5_v)
    s[data_deform_shared].compute_at(s[conv_local], rx_o_o)
    #ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused = s[data_deform_shared].fuse(ax2, ax3, ax4, ax5) # FIXME: my fuse to avoid channel
    ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused = s[data_deform_shared].fuse(ax0, ax1, ax2, ax3, ax4, ax5)
    ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_o, ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_i = s[data_deform_shared].split(ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused, factor=256)
    s[data_deform_shared].bind(ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_i, te.thread_axis("threadIdx.x"))
    s[conv].bind(n_o_o_o_f_o_o_o_fused_y_o_o_o_fused_x_o_o_o_fused, te.thread_axis("blockIdx.x"))
    s[conv].bind(n_o_o_i_f_o_o_i_fused_y_o_o_i_fused_x_o_o_i_fused, te.thread_axis("vthread"))
    s[conv].bind(n_o_i_f_o_i_fused_y_o_i_fused_x_o_i_fused, te.thread_axis("threadIdx.x"))
    n_i_f_i_y_i_x_i_fused = s[conv].fuse(n_i, f_i, y_i, x_i)
    _, n_i_f_i_y_i_x_i_fused_v = s[conv].split(n_i_f_i_y_i_x_i_fused, factor=2)
    s[conv].vectorize(n_i_f_i_y_i_x_i_fused)

    s[conv_local].pragma(n_c_o_o_o_o_f_c_o_o_o_o_fused_y_c_o_o_o_o_fused_x_c_o_o_o_o_fused, "auto_unroll_max_step", 512)
    s[conv_local].pragma(n_c_o_o_o_o_f_c_o_o_o_o_fused_y_c_o_o_o_o_fused_x_c_o_o_o_o_fused, "unroll_explicit", True)
    s[data_deform].compute_inline()
    return s
def intrin_wmma_load_matrix_A(strides_dst, strides_from, shape, layout, A_shape, C_shape, in_dtype):
    """Intrin function for loading data from shared memory to wmma.matrix_a"""
    wmma_m, wmma_n, wmma_k = shape

    A = te.placeholder(A_shape, name='A', dtype=in_dtype)
    BA = tvm.tir.decl_buffer(A.shape, A.dtype,
                             scope='shared', strides=strides_from,
                             data_alignment=32, offset_factor=8)
    C = te.compute(C_shape, lambda *i: A(*i), name='C')
    BC = tvm.tir.decl_buffer(C.shape, C.dtype,
                             scope="wmma.matrix_a", strides=strides_dst,
                             data_alignment=32, offset_factor=8)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        row = wmma_m * wmma_k
        warp_index = BC.elem_offset // row + BC.elem_offset % row // wmma_k
        ib.emit(tvm.tir.call_intrin('handle', 'tvm_load_matrix_sync',
                                    BC.data, wmma_m, wmma_n, wmma_k, warp_index,
                                    BA.access_ptr('r'), strides_from[0], layout))
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_load_matrix_W(strides_dst, strides_from, shape, layout, A_shape, C_shape, in_dtype):
    """Intrin function for loading data from shared memory to wmma.matrix_b"""
    wmma_m, wmma_n, wmma_k = shape

    A = te.placeholder(A_shape, name='A', dtype=in_dtype)
    BA = tvm.tir.decl_buffer(A.shape, A.dtype,
                             scope='shared', strides=strides_from,
                             data_alignment=32, offset_factor=8)
    C = te.compute(C_shape, lambda *i: A(*i), name='C')
    BC = tvm.tir.decl_buffer(C.shape, C.dtype,
                             scope="wmma.matrix_b", strides=strides_dst,
                             data_alignment=32, offset_factor=8)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        row = wmma_n * wmma_k
        warp_index = BC.elem_offset // row + BC.elem_offset % row // wmma_n
        ib.emit(tvm.tir.call_intrin('handle', 'tvm_load_matrix_sync',
                                    BC.data, wmma_m, wmma_n, wmma_k, warp_index,
                                    BA.access_ptr('r'), strides_from[0], layout))
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_store_matrix(strides_dst, strides_from, shape, layout, out_dtype, A_shape, C_shape):
    """Intrin function for storing the results from wmma.accumulator to shared"""
    wmma_m, wmma_n, wmma_k = shape
    A = te.placeholder(A_shape, name='A', dtype=out_dtype)
    BA = tvm.tir.decl_buffer(A.shape, A.dtype,
                             scope='wmma.accumulator',
                             strides=strides_from, data_alignment=32,
                             offset_factor=8)
    C = te.compute(C_shape, lambda *i: A(*i), name='C')
    BC = tvm.tir.decl_buffer(C.shape, C.dtype,
                             scope='shared', strides=strides_dst,
                             data_alignment=32, offset_factor=8)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        row = wmma_m * wmma_n
        warp_index = BA.elem_offset // row + BA.elem_offset % row // wmma_n
        ib.emit(tvm.tir.call_intrin('handle', 'tvm_store_matrix_sync',
                                    BA.data, wmma_m, wmma_n, wmma_k, warp_index,
                                    BC.access_ptr('w'), strides_dst[0], layout))
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(AL_gemm, WL_gemm, CL_compute, strides_A,
                     strides_W, strides_Conv, shape):
    """Intrin for wmma fill_fragment and mma_sync
    Parameters
    ----------
    AL_gemm : tvm.te.placeholder
        wmma matrix A
    WL_gemm : tvm.te.placeholder
        wmma matrix B
    CL_compute : tvm.te.compute
        The definition of wmma gemm
    """
    wmma_m, wmma_n, wmma_k = shape
    A = AL_gemm
    B = WL_gemm
    C = CL_compute

    BA = tvm.tir.decl_buffer(A.shape, A.dtype, name='BA',
                             scope='wmma.matrix_a', data_alignment=32,
                             offset_factor=8, strides=strides_A)
    BB = tvm.tir.decl_buffer(B.shape, B.dtype, name='BB',
                             scope='wmma.matrix_b', data_alignment=32,
                             offset_factor=8, strides=strides_W)
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, name='BC',
                             scope='wmma.accumulator', data_alignment=32,
                             offset_factor=8, strides=strides_Conv)

    def intrin_func(ins, outs):
        BA, BB = ins
        BC, = outs

        def warp_idnex(offset, row, col):
            row = row * col
            return offset // row + offset % row // col

        warp_index_A = warp_idnex(BA.elem_offset, wmma_m, wmma_k)
        warp_index_B = warp_idnex(BB.elem_offset, wmma_k, wmma_n)
        warp_index_C = warp_idnex(BC.elem_offset, wmma_m, wmma_n)

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin('handle', 'tvm_fill_fragment', BC.data, wmma_m, wmma_n, wmma_k,
                                    warp_index_C, 0.0))
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_intrin('handle', 'tvm_mma_sync',
                                        BC.data, warp_index_C,
                                        BA.data, warp_index_A,
                                        BB.data, warp_index_B,
                                        BC.data, warp_index_C))
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC}) 
