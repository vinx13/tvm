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
#pylint: disable=unused-argument
"""Automatic quantization toolkit."""
from __future__ import absolute_import
import numpy as np
import os
import pickle

from . import _quantize
from .. import expr as _expr
from .. import build_module as _build
from .. import module as _module
from .. import ir_pass as _ir_pass
from .. import transform as _transform
from .. import op as _op
from ... import make as _make
from ...contrib import graph_runtime
from ... import context
from ..base import NodeBase, register_relay_node


load_scale = False

class QAnnotateKind(object):
    """Denote the kind of annotation field, corresponding
    to different nbit configure."""
    INPUT = 1
    WEIGHT = 2
    ACTIVATION = 3
    BIAS = 4


def kind2str(kind):
    """Convert a `QAnnotateKind` to string"""
    str_map = {
        QAnnotateKind.INPUT: "input",
        QAnnotateKind.WEIGHT: "weight",
        QAnnotateKind.ACTIVATION: "activation",
        QAnnotateKind.BIAS: "bias",
    }
    assert kind in str_map
    return str_map[kind]


@register_relay_node("relay.quantize.QConfig")
class QConfig(NodeBase):
    """Configure the quantization behavior by setting config variables.

    Note
    ----
    This object is backed by node system in C++, with arguments that can be
    exchanged between python and C++.

    Do not construct directly, use qconfig instead.

    The fields that are backed by the C++ node are immutable once an instance
    is constructed. See _node_defaults for the fields.
    """

    _node_defaults = {
        "nbit_input": 8,
        "nbit_weight": 8,
        "nbit_activation": 32,
        "nbit_bias": 32,
        "dtype_input": "int8",
        "dtype_weight": "int8",
        "dtype_activation": "int32",
        "dtype_bias": "int32",
        "global_scale": 8.0,
        "skip_k_conv": 1,
        "skip_conv_layers": None,
        "round_for_shift": True,
        "store_lowbit_output": True,
        "debug_enabled_ops": None,
        "use_stop_fusion": True
    }

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        super(QConfig, self).__init__(handle)
        self.handle = handle

    def guard(self, ref_call):
        op_name = ref_call.op.name
        if self.debug_enabled_ops is not None:
            name_list = [x.value for x in self.debug_enabled_ops]
            if op_name not in name_list:
                return False
        return True

    def get_nbit_by_kind(self, kind):
        name = kind2str(kind)
        return getattr(self, 'nbit_' + name)

    def get_dtype_by_kind(self, kind):
        name = kind2str(kind)
        return getattr(self, 'dtype_' + name)

    def __enter__(self):
        # pylint: disable=protected-access
        _quantize._EnterQConfigScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _quantize._ExitQConfigScope(self)

    def __setattr__(self, name, value):
        if name in QConfig._node_defaults:
            raise AttributeError(
                "'%s' object cannot set attribute '%s'" % (str(type(self)), name))
        return super(QConfig, self).__setattr__(name, value)


def current_qconfig():
    """Get the current quantization configuration."""
    return _quantize._GetCurrentQConfig()


def qconfig(**kwargs):
    """Configure the quantization behavior by setting config variables.

    Parameters
    ---------
    nbit_dict: dict of QAnnotateKind -> int
        Number of bit for every kind of annotate field.

    global_scale: float
        The global scale for calibration.

    skip_k_conv: int
        The number of skipped conv2d.

    skip_conv_layers: list
        Different way of specifying which layers to avoid. Provide a list of indices
        that indicate which conv2d layers to leave untouched.

    round_for_shift: boolean
        Whether to add bias for rounding during shift.

    store_lowbit_output: boolean
        Whether to store low-bit integer back as output before dequantizing.
        Some accelerators need this, e.g. VTA.

    use_stop_fusion: boolean
        Whether add stop_fusion when casting to dtype_activation. stop_fusion forces lowbit
        results to be stored in memory.

    Returns
    -------
    config: QConfig
        The quantization configuration
    """
    node_args = {k: v if k not in kwargs else kwargs[k]
                 for k, v in QConfig._node_defaults.items()}
    return _make.node("relay.quantize.QConfig", **node_args)


CONV_COUNTER = 0


def _conv_counter():
    """Get the global counter for conv2d."""
    return CONV_COUNTER


def _set_conv_counter(n):
    """Set the value of the global conv2d counter."""
    global CONV_COUNTER
    CONV_COUNTER = n


def arr_hist(arrs):
    arr = np.concatenate(arrs).reshape(-1)
    min_val = np.min(arr)
    max_val = np.max(arr)
    th = max(abs(min_val), abs(max_val))
    num_bins = 8001
    hist, edges = np.histogram(arr, bins=num_bins, range=(-th, th))
    return (hist, edges, min_val, max_val)


def collect_stats(graph, dataset):
    if os.path.exists('histogram.pkl'):
        with open('histogram.pkl', 'rb') as f:
            return pickle.load(f)

    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    quantized_exprs = []

    def visit_func(expr):
        """Internal visit function"""
        if isinstance(expr, _expr.Call) and expr.op == quantize_op and expr.attrs.kind not in [QAnnotateKind.WEIGHT, QAnnotateKind.BIAS]:
            quantized_exprs.append(expr.args[0])

    _ir_pass.post_order_visit(graph, visit_func)
    if len(quantized_exprs) == 0:
        return []
    graph = _expr.Function(graph.params, _expr.Tuple(quantized_exprs))

    graph_json, lib, params = _build.build(graph, 'cuda')
    module = graph_runtime.create(graph_json, lib, context('cuda', 0))
    module.set_input(**params)

    num_outputs = module.get_num_outputs()
    outputs = [[] for i in range(num_outputs)]

    for batch_id, batch in enumerate(dataset):
        print('batch {}..'.format(batch_id))
        module.set_input(**batch)
        module.run()
        for i in range(num_outputs):
            output = module.get_output(i).asnumpy()
            outputs[i].append(output)

    result = list(map(arr_hist, outputs))
    with open('histogram.pkl', 'wb') as f:
        pickle.dump(result, f)
    return result


from scipy import stats
def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


def _get_optimal_threshold(profile, num_bins=8001, num_quantized_bins=255):
    """Given a dataset, find the optimal threshold for quantizing it.
    The reference distribution is `q`, and the candidate distribution is `p`.
    `q` is a truncated version of the original distribution.

    Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    """
    #assert isinstance(arr, np.ndarray)
    hist, hist_edges, min_val, max_val = profile
    th = max(abs(min_val), abs(max_val))

    zero_bin_idx = num_bins // 2
    num_half_quantized_bins = num_quantized_bins // 2
    assert np.allclose(hist_edges[zero_bin_idx] + hist_edges[zero_bin_idx + 1],
                       0, rtol=1e-5, atol=1e-7)

    thresholds = np.zeros(num_bins // 2 + 1 - num_quantized_bins // 2)
    divergence = np.zeros_like(thresholds)
    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    # i means the number of bins on half axis excluding the zero bin.
    for i in range(num_quantized_bins // 2,
                   num_bins // 2 + 1):
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop]
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        assert p.size % 2 == 1
        assert p.size >= num_quantized_bins
        # put left outlier count in p[0]
        left_outlier_count = np.sum(hist[0:p_bin_idx_start])
        p[0] += left_outlier_count
        # put right outlier count in p[-1]
        right_outlier_count = np.sum(hist[p_bin_idx_stop:])
        p[-1] += right_outlier_count
        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int32)

        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // num_quantized_bins
        # merge hist into num_quantized_bins bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = len(is_nonzeros)
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        p = _smooth_distribution(p)
        # There is a chance that q is an invalid probability distribution.
        try:
            q = _smooth_distribution(q)
        except ValueError:
            divergence[i - num_half_quantized_bins] = float("inf")
        divergence[i - num_half_quantized_bins] = stats.entropy(p, q)

    min_divergence_idx = np.argmin(divergence)
    min_divergence = divergence[min_divergence_idx]
    opt_th = thresholds[min_divergence_idx]
    return min_val, max_val, min_divergence, opt_th
# pylint: enable=line-too-long

def power2_scale(arr):
    """calculate weight scale with nearest mode-2 scale"""
    if not isinstance(arr, np.ndarray):
        arr = arr.asnumpy()
    val = np.amax(np.abs(arr))
    return 2**np.math.ceil(np.math.log(val, 2)) if val > 0 else 1.0


def act_power2_scale(profile):
    _, _, min_val, max_val = profile
    th = max(abs(min_val), abs(max_val))
    return 2**np.math.ceil(np.math.log(th, 2)) if th > 0 else 1.0

def max_scale(arr):
    if not isinstance(arr, np.ndarray):
        arr = arr.asnumpy()
    val = np.amax(np.abs(arr))
    return val

def act_kld(profile):
    _, _, _, val = _get_optimal_threshold(profile, num_bins=8001, num_quantized_bins=255)
    return val if val > 0 else 1.0
    # round kld result to power-of-2
    # return 2**np.math.ceil(np.math.log(val, 2)) if val > 0 else 1.0

def act_global(profile):
    return 8.0


def calibrate(graph, mod=None, ctx=None, dataset=None):
    """The calibrate procedure will try to calculate the content of
    dom_scale, nbit, clip_min, clip_max for every `simulated_quantize`
    operator.

    Parameters
    ---------
    graph: Function
        The simulation graph after annotation.

    mod: tvm.relay.Module
        The module where calibration happens on.

    ctx: tvm.relay.PassContext
        The pass context used for calibration.

    Returns
    -------
    ret: Function
        The graph after calibration
    """

    #fcalib_act = act_power2_scale
    #fcalib_act = act_global
    fcalib_act = act_kld
    #fcalib_weight = power2_scale
    fcalib_weight = max_scale

    cfg = current_qconfig()
    const_params = {}
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")

    outputs = None
    scales = None

    counter = [0]

    def visit_func(expr):
        """Internal visit function"""
        if isinstance(expr, _expr.Call) and expr.op == quantize_op:
            _, ndom_scale, nclip_min, nclip_max = expr.args
            attrs = expr.attrs
            kind = attrs.kind
            nbit = cfg.get_nbit_by_kind(kind)

            valid_bit = nbit - attrs.sign

            if kind in [QAnnotateKind.WEIGHT, QAnnotateKind.BIAS]:
                if outputs is not None:
                    # this is the second time to reach here, scales have been calculated
                    return
                var = expr.args[0]
                assert isinstance(var, _expr.Constant)
                scale = fcalib_weight(var.data)
                print('weight scale: {}'.format(scale))
            else:
                if outputs is not None:
                    scale = scales[counter[0]]
                    counter[0] += 1
                    print('{} / {} ...'.format(counter[0], len(outputs)))
                    print('act scale: {}'.format(scale))
                else:
                    scale = cfg.global_scale

            def _make_const(val):
                return _expr.const(val, 'float32')

            valid_range = 2**valid_bit
            const_params[ndom_scale] = _make_const(scale / valid_range)
            if kind in [QAnnotateKind.BIAS]:
                const_params[ndom_scale] = _make_const(scale / (2**15))
            const_params[nclip_min] = _make_const(- (valid_range - 1))
            const_params[nclip_max] = _make_const((valid_range - 1))

    _ir_pass.post_order_visit(graph, visit_func)
    original_graph = graph
    graph = _expr.bind(original_graph, const_params)

    if dataset is not None:
        global load_scale

        print('Calibrating on dataset')
        outputs = collect_stats(graph, dataset)

        if load_scale:
            with open('scale.pkl', 'rb') as f:
                scales = pickle.load(f)
            print(scales)
            # for i in range(len(scales)):
            #     scales[i] = 2**np.math.ceil(np.math.log(scales[i], 2))
            # print(scales)
        else:
            scales = []
            for profile in outputs:
                print(len(scales)+1)
                scales.append(fcalib_act(profile))

            print('scales')
            print(scales)
            with open('scale.pkl', 'wb') as f:
                pickle.dump(scales, f)

        _ir_pass.post_order_visit(original_graph, visit_func)
        assert counter[0] == len(outputs)
        graph = _expr.bind(original_graph, const_params)

    return graph


def annotate():
    """Given a float32 graph, this pass will rewrite the graph and return
    a graph which simulates the error brought by the current quantization
    scheme.

    Returns
    -------
    ret: tvm.relay.Pass
        The registered pass for quantization annotation.
    """
    return _quantize.QuantizeAnnotate()


def realize():
    """The realize pass will transform the simulated quantized graph, which
    actually computes with float32, to a real low-bit integer graph. It will
    replace the `simulated_quantize` with several fine-grained operators like
    add, multiply, and shift as much as possible for better performance.

    Returns
    -------
    ret: tvm.relay.Pass
        The registered pass for quantization realization.
    """
    return _quantize.QuantizeRealize()


def _bind_params(func, params):
    """Bind the params to the expression.
    """
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = _expr.const(v)
    return _expr.bind(func, bind_dict)


def quantize(graph, params=None, dataset=None):
    """ The quantization procedure. Before running the three main
    procedure of quantization, "annotate", "calibrate" and "realize"
    , we need to do "SimplifyInference", "FoldScaleAxis", "FoldConstant"
    first for optimizing.

    Parameters
    ---------
    graph: Function
        The original graph.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    dataset: list of dict of Var -> NDArray
        The calibration dataset.

    Returns
    -------
    ret: Function
        The graph after quantization
    """
    if params:
        graph = _bind_params(graph, params)

    mod = _module.Module.from_expr(graph)
    # Perform "SimplifyInference", "FoldScaleAxis", "FoldConstant", and
    # "CanonicalizeOps" optimization before quantization.
    optimize = _transform.Sequential([_transform.SimplifyInference(),
                                      _transform.FoldConstant(),
                                      _transform.FoldScaleAxis(),
                                      _transform.CanonicalizeOps(),
                                      _transform.FoldConstant()])

    def _calibrate(*args, **kwargs):
        return calibrate(*args, dataset=dataset, **kwargs)
    calibrate_pass = _transform.function_pass(_calibrate, opt_level=1,
                                              name="QuantizeCalibrate")
    _set_conv_counter(0)  # reset counter
    quantize_seq = _transform.Sequential([annotate(),
                                          calibrate_pass,
                                          realize(),
                                          _transform.FoldConstant()])
    with _transform.PassContext(opt_level=3,
                                required_pass=["QuantizeAnnotate",
                                               "QuantizeCalibrate",
                                               "QuantizeRealize"]):
        mod = optimize(mod)
        mod = quantize_seq(mod)
    return mod[mod.entry_func.name_hint]
