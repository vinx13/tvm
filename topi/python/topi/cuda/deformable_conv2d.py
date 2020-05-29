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
from ..util import traverse_inline


@autotvm.register_topi_compute("deformable_conv2d_nchw.cuda")
def deformable_conv2d_nchw(cfg, data, offset, kernel, strides, padding, dilation,
                           deformable_groups, groups, out_dtype):
    return nn.deformable_conv2d_nchw(data, offset, kernel, strides, padding, dilation,
                                     deformable_groups, groups, out_dtype)

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
            _schedule_manual_cuda(s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_direct_cuda(cfg, s, conv):
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
    if conv.op in s.outputs:
        # output is conv
        conv_local = s.cache_write(conv, 'local')
    else:
        s[conv].set_scope('local')
        conv_local = conv
        # output is relu but named conv
        conv = s.outputs[0].output(0)

    padded_data = data_deform.op.input_tensors[0]

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
    ax0_ax1_fused_ax2_fused_ax3_fused_o, ax0_ax1_fused_ax2_fused_ax3_fused_i = s[kernel_shared].split(ax0_ax1_fused_ax2_fused_ax3_fused, factor=256)
    s[kernel_shared].bind(ax0_ax1_fused_ax2_fused_ax3_fused_i, te.thread_axis("threadIdx.x"))
    data_deform_shared = s.cache_read(data_deform, "shared", [conv_local])
    ax0, ax1, ax2, ax3, ax4, ax5 = tuple(data_deform_shared.op.axis)
    s[data_deform_shared].compute_at(s[conv_local], rx_o_o)
    ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused = s[data_deform_shared].fuse(ax0, ax1, ax2, ax3, ax4, ax5)
    ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_o, ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_i = s[data_deform_shared].split(ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused, factor=256)
    s[data_deform_shared].bind(ax0_ax1_fused_ax2_fused_ax3_fused_ax4_fused_ax5_fused_i, te.thread_axis("threadIdx.x"))
    s[conv].bind(n_o_o_o_f_o_o_o_fused_y_o_o_o_fused_x_o_o_o_fused, te.thread_axis("blockIdx.x"))
    s[conv].bind(n_o_o_i_f_o_o_i_fused_y_o_o_i_fused_x_o_o_i_fused, te.thread_axis("vthread"))
    s[conv].bind(n_o_i_f_o_i_fused_y_o_i_fused_x_o_i_fused, te.thread_axis("threadIdx.x"))
    s[conv_local].pragma(n_c_o_o_o_o_f_c_o_o_o_o_fused_y_c_o_o_o_o_fused_x_c_o_o_o_o_fused, "auto_unroll_max_step", 512)
    s[conv_local].pragma(n_c_o_o_o_o_f_c_o_o_o_o_fused_y_c_o_o_o_o_fused_x_c_o_o_o_o_fused, "unroll_explicit", True)
    s[data_deform].compute_inline()

    return s
    
