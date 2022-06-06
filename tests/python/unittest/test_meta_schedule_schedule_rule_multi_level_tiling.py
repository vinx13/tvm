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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import tvm
import tvm.testing
from tvm import te
from tvm.meta_schedule import schedule_rule
from tvm.meta_schedule.space_generator.post_order_apply import PostOrderApply
from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.schedule_rule import (
    auto_inline,
    multi_level_tiling,
    multi_level_tiling_tensor_core,
)
from tvm.meta_schedule.testing.space_generation import check_trace
from tvm.meta_schedule.tune_context import TuneContext
from tvm.script import tir as T
from tvm.target import Target
from tvm.te import create_prim_func
from tvm.tir.tensor_intrin import DP4A_INTRIN
from tvm.tir.tensor_intrin import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN


def _create_context(mod, target, rule) -> TuneContext:
    if not isinstance(rule, (list, tuple)):
        rule = [rule]
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=rule,
        task_name="test",
    )
    return ctx


def test_cpu_matmul():
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7], preserve_unit_iters=True)",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15], preserve_unit_iters=True)",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21], preserve_unit_iters=True)",
            "sch.reorder(l8, l16, l9, l17, l22, l10, l18, l23, l11, l19)",
            'b24 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")',
            "sch.reverse_compute_at(block=b24, loop=l17, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7], preserve_unit_iters=True)",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15], preserve_unit_iters=True)",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21], preserve_unit_iters=True)",
            "sch.reorder(l8, l16, l9, l17, l22, l10, l18, l23, l11, l19)",
            'b24 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")',
            "sch.reverse_compute_at(block=b24, loop=l16, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7], preserve_unit_iters=True)",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15], preserve_unit_iters=True)",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21], preserve_unit_iters=True)",
            "sch.reorder(l8, l16, l9, l17, l22, l10, l18, l23, l11, l19)",
        ],
    ]
    target = Target("llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_cpu_matmul_relu():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7], preserve_unit_iters=True)",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15], preserve_unit_iters=True)",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21], preserve_unit_iters=True)",
            "sch.reorder(l8, l16, l9, l17, l22, l10, l18, l23, l11, l19)",
            "b24, = sch.get_consumers(block=b0)",
            "sch.reverse_compute_at(block=b24, loop=l17, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7], preserve_unit_iters=True)",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15], preserve_unit_iters=True)",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21], preserve_unit_iters=True)",
            "sch.reorder(l8, l16, l9, l17, l22, l10, l18, l23, l11, l19)",
            "b24, = sch.get_consumers(block=b0)",
            "sch.reverse_compute_at(block=b24, loop=l16, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7], preserve_unit_iters=True)",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15], preserve_unit_iters=True)",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21], preserve_unit_iters=True)",
            "sch.reorder(l8, l16, l9, l17, l22, l10, l18, l23, l11, l19)",
        ],
    ]
    # pylint: enable=line-too-long
    target = Target("llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_relu(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_cuda_matmul():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l1, n=5, max_innermost_factor=64)",
            "l9, l10, l11, l12, l13 = sch.split(loop=l1, factors=[v4, v5, v6, v7, v8], preserve_unit_iters=True)",
            "v14, v15, v16, v17, v18 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64)",
            "l19, l20, l21, l22, l23 = sch.split(loop=l2, factors=[v14, v15, v16, v17, v18], preserve_unit_iters=True)",
            "v24, v25, v26 = sch.sample_perfect_tile(loop=l3, n=3, max_innermost_factor=64)",
            "l27, l28, l29 = sch.split(loop=l3, factors=[v24, v25, v26], preserve_unit_iters=True)",
            "sch.reorder(l9, l19, l10, l20, l11, l21, l27, l28, l12, l22, l29, l13, l23)",
            "l30 = sch.fuse(l9, l19, preserve_unit_iters=True)",
            'sch.bind(loop=l30, thread_axis="blockIdx.x")',
            "l31 = sch.fuse(l10, l20, preserve_unit_iters=True)",
            'sch.bind(loop=l31, thread_axis="vthread.x")',
            "l32 = sch.fuse(l11, l21, preserve_unit_iters=True)",
            'sch.bind(loop=l32, thread_axis="threadIdx.x")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)',
            'b33 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")',
            "sch.reverse_compute_at(block=b33, loop=l32, preserve_unit_loops=True)",
            'b34 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared")',
            "sch.compute_at(block=b34, loop=l27, preserve_unit_loops=True)",
            "l35, l36, l37, l38, l39, l40 = sch.get_loops(block=b34)",
            "l41 = sch.fuse(l39, l40, preserve_unit_iters=True)",
            "v42 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v42)',
            'b43 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b43, loop=l27, preserve_unit_loops=True)",
            "l44, l45, l46, l47, l48, l49 = sch.get_loops(block=b43)",
            "l50 = sch.fuse(l48, l49, preserve_unit_iters=True)",
            "v51 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b43, ann_key="meta_schedule.cooperative_fetch", ann_val=v51)',
        ]
    ]
    # pylint: enable=line-too-long
    target = Target("cuda --max_threads_per_block=1024 --thread_warp_size=32", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_cuda_matmul_relu():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l1, n=5, max_innermost_factor=64)",
            "l9, l10, l11, l12, l13 = sch.split(loop=l1, factors=[v4, v5, v6, v7, v8], preserve_unit_iters=True)",
            "v14, v15, v16, v17, v18 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64)",
            "l19, l20, l21, l22, l23 = sch.split(loop=l2, factors=[v14, v15, v16, v17, v18], preserve_unit_iters=True)",
            "v24, v25, v26 = sch.sample_perfect_tile(loop=l3, n=3, max_innermost_factor=64)",
            "l27, l28, l29 = sch.split(loop=l3, factors=[v24, v25, v26], preserve_unit_iters=True)",
            "sch.reorder(l9, l19, l10, l20, l11, l21, l27, l28, l12, l22, l29, l13, l23)",
            "l30 = sch.fuse(l9, l19, preserve_unit_iters=True)",
            'sch.bind(loop=l30, thread_axis="blockIdx.x")',
            "l31 = sch.fuse(l10, l20, preserve_unit_iters=True)",
            'sch.bind(loop=l31, thread_axis="vthread.x")',
            "l32 = sch.fuse(l11, l21, preserve_unit_iters=True)",
            'sch.bind(loop=l32, thread_axis="threadIdx.x")',
            'b33 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")',
            "sch.reverse_compute_at(block=b33, loop=l32, preserve_unit_loops=True)",
            'b34 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared")',
            "sch.compute_at(block=b34, loop=l27, preserve_unit_loops=True)",
            "l35, l36, l37, l38, l39, l40 = sch.get_loops(block=b34)",
            "l41 = sch.fuse(l39, l40, preserve_unit_iters=True)",
            "v42 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v42)',
            'b43 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b43, loop=l27, preserve_unit_loops=True)",
            "l44, l45, l46, l47, l48, l49 = sch.get_loops(block=b43)",
            "l50 = sch.fuse(l48, l49, preserve_unit_iters=True)",
            "v51 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b43, ann_key="meta_schedule.cooperative_fetch", ann_val=v51)',
        ]
    ]
    # pylint: enable=line-too-long
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_relu(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_cuda_sum_with_trivial_block_iter():
    @T.prim_func
    def sum_with_trivial_block_iter(
        A: T.Buffer[(1, 64, 768), "float32"], B: T.Buffer[(1, 64, 1), "float32"]
    ) -> None:
        for i0, i1, i2, i3 in T.grid(1, 64, 1, 768):
            with T.block("sum"):
                ax0, ax1, ax2, k2 = T.axis.remap("SSSR", [i0, i1, i2, i3])
                T.reads(A[ax0, ax1, k2])
                T.writes(B[ax0, ax1, ax2])
                with T.init():
                    B[ax0, ax1, ax2] = T.float32(0)
                B[ax0, ax1, ax2] = B[ax0, ax1, ax2] + A[ax0, ax1, k2]

    # Expect nothing to happen - the rule is not supposed to be applied in this case
    expected = [[]]
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        sum_with_trivial_block_iter,
        target=target,
        rule=multi_level_tiling(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


@tvm.script.ir_module
class Conv2dNCHWcVNNIModule:
    @T.prim_func
    def main(
        placeholder: T.Buffer[(1, 4, 56, 56, 16), "uint8"],
        placeholder_1: T.Buffer[(16, 4, 1, 1, 4, 16, 4), "int8"],
        conv2d_NCHWc_int8: T.Buffer[(1, 16, 56, 56, 16), "int32"],
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 in T.grid(1, 16, 56, 56, 16, 1, 1, 4, 4, 4):
            with T.block("conv2d_NCHWc_int8"):
                (
                    n,
                    oc_chunk,
                    oh,
                    ow,
                    oc_block,
                    kh,
                    kw,
                    ic_outer,
                    ic_f_inner,
                    ic_s_inner,
                ) = T.axis.remap("SSSSSRRRRR", [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9])
                T.reads(
                    placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner],
                    placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner],
                )
                T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block])
                with T.init():
                    conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = 0
                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_int8[
                    n, oc_chunk, oh, ow, oc_block
                ] + T.cast(
                    placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner], "int32"
                ) * T.cast(
                    placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner],
                    "int32",
                )


def test_multi_level_tiling_conv2d_nchwc_vnni():
    target = "llvm -mcpu=cascadelake -num-cores 4"
    ctx = _create_context(
        Conv2dNCHWcVNNIModule,
        target=tvm.target.Target(target),
        rule=schedule_rule.MultiLevelTilingWithIntrin(
            VNNI_INTRIN,
            structure="SSRSRS",
            tile_binds=None,
            max_innermost_factor=64,
            vector_load_lens=None,
            reuse_read=None,
            reuse_write=schedule_rule.ReuseType(
                req="may",
                levels=[1, 2],
                scope="global",
            ),
        ),
    )

    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)

    expected = [
        """b0 = sch.get_block(name="conv2d_NCHWc_int8", func_name="main")
sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = sch.get_loops(block=b0)
l11, l12 = sch.split(loop=l10, factors=[1, 4], preserve_unit_iters=True)
l13, l14 = sch.split(loop=l5, factors=[1, 16], preserve_unit_iters=True)
l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26 = sch.get_loops(block=b0)
sch.reorder(l21, l22, l23, l24, l25, l14, l12)
b27 = sch.blockize(loop=l14)
sch.annotate(block_or_loop=b27, ann_key="meta_schedule.auto_tensorize", ann_val="dot_16x4_vnni")
l28, l29, l30, l31, l32, l33, l34, l35, l36, l37 = sch.get_loops(block=b27)
v38, v39, v40, v41 = sch.sample_perfect_tile(loop=l28, n=4, max_innermost_factor=64)
l42, l43, l44, l45 = sch.split(loop=l28, factors=[v38, v39, v40, v41], preserve_unit_iters=True)
v46, v47, v48, v49 = sch.sample_perfect_tile(loop=l29, n=4, max_innermost_factor=64)
l50, l51, l52, l53 = sch.split(loop=l29, factors=[v46, v47, v48, v49], preserve_unit_iters=True)
v54, v55, v56, v57 = sch.sample_perfect_tile(loop=l30, n=4, max_innermost_factor=64)
l58, l59, l60, l61 = sch.split(loop=l30, factors=[v54, v55, v56, v57], preserve_unit_iters=True)
v62, v63, v64, v65 = sch.sample_perfect_tile(loop=l31, n=4, max_innermost_factor=64)
l66, l67, l68, l69 = sch.split(loop=l31, factors=[v62, v63, v64, v65], preserve_unit_iters=True)
v70, v71, v72, v73 = sch.sample_perfect_tile(loop=l32, n=4, max_innermost_factor=64)
l74, l75, l76, l77 = sch.split(loop=l32, factors=[v70, v71, v72, v73], preserve_unit_iters=True)
v78, v79 = sch.sample_perfect_tile(loop=l33, n=2, max_innermost_factor=64)
l80, l81 = sch.split(loop=l33, factors=[v78, v79], preserve_unit_iters=True)
v82, v83 = sch.sample_perfect_tile(loop=l34, n=2, max_innermost_factor=64)
l84, l85 = sch.split(loop=l34, factors=[v82, v83], preserve_unit_iters=True)
v86, v87 = sch.sample_perfect_tile(loop=l35, n=2, max_innermost_factor=64)
l88, l89 = sch.split(loop=l35, factors=[v86, v87], preserve_unit_iters=True)
v90, v91 = sch.sample_perfect_tile(loop=l36, n=2, max_innermost_factor=64)
l92, l93 = sch.split(loop=l36, factors=[v90, v91], preserve_unit_iters=True)
v94, v95 = sch.sample_perfect_tile(loop=l37, n=2, max_innermost_factor=64)
l96, l97 = sch.split(loop=l37, factors=[v94, v95], preserve_unit_iters=True)
sch.reorder(l42, l50, l58, l66, l74, l43, l51, l59, l67, l75, l80, l84, l88, l92, l96, l44, l52, l60, l68, l76, l81, l85, l89, l93, l97, l45, l53, l61, l69, l77)
b98 = sch.cache_write(block=b27, write_buffer_index=0, storage_scope="global")
sch.reverse_compute_at(block=b98, loop=l75, preserve_unit_loops=True)""".split(
            "\n"
        ),
        """b0 = sch.get_block(name="conv2d_NCHWc_int8", func_name="main")
sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = sch.get_loops(block=b0)
l11, l12 = sch.split(loop=l10, factors=[1, 4], preserve_unit_iters=True)
l13, l14 = sch.split(loop=l5, factors=[1, 16], preserve_unit_iters=True)
l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26 = sch.get_loops(block=b0)
sch.reorder(l21, l22, l23, l24, l25, l14, l12)
b27 = sch.blockize(loop=l14)
sch.annotate(block_or_loop=b27, ann_key="meta_schedule.auto_tensorize", ann_val="dot_16x4_vnni")
l28, l29, l30, l31, l32, l33, l34, l35, l36, l37 = sch.get_loops(block=b27)
v38, v39, v40, v41 = sch.sample_perfect_tile(loop=l28, n=4, max_innermost_factor=64)
l42, l43, l44, l45 = sch.split(loop=l28, factors=[v38, v39, v40, v41], preserve_unit_iters=True)
v46, v47, v48, v49 = sch.sample_perfect_tile(loop=l29, n=4, max_innermost_factor=64)
l50, l51, l52, l53 = sch.split(loop=l29, factors=[v46, v47, v48, v49], preserve_unit_iters=True)
v54, v55, v56, v57 = sch.sample_perfect_tile(loop=l30, n=4, max_innermost_factor=64)
l58, l59, l60, l61 = sch.split(loop=l30, factors=[v54, v55, v56, v57], preserve_unit_iters=True)
v62, v63, v64, v65 = sch.sample_perfect_tile(loop=l31, n=4, max_innermost_factor=64)
l66, l67, l68, l69 = sch.split(loop=l31, factors=[v62, v63, v64, v65], preserve_unit_iters=True)
v70, v71, v72, v73 = sch.sample_perfect_tile(loop=l32, n=4, max_innermost_factor=64)
l74, l75, l76, l77 = sch.split(loop=l32, factors=[v70, v71, v72, v73], preserve_unit_iters=True)
v78, v79 = sch.sample_perfect_tile(loop=l33, n=2, max_innermost_factor=64)
l80, l81 = sch.split(loop=l33, factors=[v78, v79], preserve_unit_iters=True)
v82, v83 = sch.sample_perfect_tile(loop=l34, n=2, max_innermost_factor=64)
l84, l85 = sch.split(loop=l34, factors=[v82, v83], preserve_unit_iters=True)
v86, v87 = sch.sample_perfect_tile(loop=l35, n=2, max_innermost_factor=64)
l88, l89 = sch.split(loop=l35, factors=[v86, v87], preserve_unit_iters=True)
v90, v91 = sch.sample_perfect_tile(loop=l36, n=2, max_innermost_factor=64)
l92, l93 = sch.split(loop=l36, factors=[v90, v91], preserve_unit_iters=True)
v94, v95 = sch.sample_perfect_tile(loop=l37, n=2, max_innermost_factor=64)
l96, l97 = sch.split(loop=l37, factors=[v94, v95], preserve_unit_iters=True)
sch.reorder(l42, l50, l58, l66, l74, l43, l51, l59, l67, l75, l80, l84, l88, l92, l96, l44, l52, l60, l68, l76, l81, l85, l89, l93, l97, l45, l53, l61, l69, l77)
b98 = sch.cache_write(block=b27, write_buffer_index=0, storage_scope="global")
sch.reverse_compute_at(block=b98, loop=l74, preserve_unit_loops=True)""".split(
            "\n"
        ),
        """b0 = sch.get_block(name="conv2d_NCHWc_int8", func_name="main")
sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = sch.get_loops(block=b0)
l11, l12 = sch.split(loop=l10, factors=[1, 4], preserve_unit_iters=True)
l13, l14 = sch.split(loop=l5, factors=[1, 16], preserve_unit_iters=True)
l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26 = sch.get_loops(block=b0)
sch.reorder(l21, l22, l23, l24, l25, l14, l12)
b27 = sch.blockize(loop=l14)
sch.annotate(block_or_loop=b27, ann_key="meta_schedule.auto_tensorize", ann_val="dot_16x4_vnni")
l28, l29, l30, l31, l32, l33, l34, l35, l36, l37 = sch.get_loops(block=b27)
v38, v39, v40, v41 = sch.sample_perfect_tile(loop=l28, n=4, max_innermost_factor=64)
l42, l43, l44, l45 = sch.split(loop=l28, factors=[v38, v39, v40, v41], preserve_unit_iters=True)
v46, v47, v48, v49 = sch.sample_perfect_tile(loop=l29, n=4, max_innermost_factor=64)
l50, l51, l52, l53 = sch.split(loop=l29, factors=[v46, v47, v48, v49], preserve_unit_iters=True)
v54, v55, v56, v57 = sch.sample_perfect_tile(loop=l30, n=4, max_innermost_factor=64)
l58, l59, l60, l61 = sch.split(loop=l30, factors=[v54, v55, v56, v57], preserve_unit_iters=True)
v62, v63, v64, v65 = sch.sample_perfect_tile(loop=l31, n=4, max_innermost_factor=64)
l66, l67, l68, l69 = sch.split(loop=l31, factors=[v62, v63, v64, v65], preserve_unit_iters=True)
v70, v71, v72, v73 = sch.sample_perfect_tile(loop=l32, n=4, max_innermost_factor=64)
l74, l75, l76, l77 = sch.split(loop=l32, factors=[v70, v71, v72, v73], preserve_unit_iters=True)
v78, v79 = sch.sample_perfect_tile(loop=l33, n=2, max_innermost_factor=64)
l80, l81 = sch.split(loop=l33, factors=[v78, v79], preserve_unit_iters=True)
v82, v83 = sch.sample_perfect_tile(loop=l34, n=2, max_innermost_factor=64)
l84, l85 = sch.split(loop=l34, factors=[v82, v83], preserve_unit_iters=True)
v86, v87 = sch.sample_perfect_tile(loop=l35, n=2, max_innermost_factor=64)
l88, l89 = sch.split(loop=l35, factors=[v86, v87], preserve_unit_iters=True)
v90, v91 = sch.sample_perfect_tile(loop=l36, n=2, max_innermost_factor=64)
l92, l93 = sch.split(loop=l36, factors=[v90, v91], preserve_unit_iters=True)
v94, v95 = sch.sample_perfect_tile(loop=l37, n=2, max_innermost_factor=64)
l96, l97 = sch.split(loop=l37, factors=[v94, v95], preserve_unit_iters=True)
sch.reorder(l42, l50, l58, l66, l74, l43, l51, l59, l67, l75, l80, l84, l88, l92, l96, l44, l52, l60, l68, l76, l81, l85, l89, l93, l97, l45, l53, l61, l69, l77)""".split(
            "\n"
        ),
    ]

    check_trace(spaces, expected)


def test_multi_level_tiling_dense_dpa4():
    m, n, k = 128, 128, 128

    X = te.placeholder((m, k), name="X", dtype="int8")
    W = te.placeholder((n, k), name="W", dtype="int8")
    ak = te.reduce_axis((0, k), name="k")

    matmul = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype("int32") * W[j, ak].astype("int32"),
            axis=ak,
        ),
        name="compute",
    )

    func = te.create_prim_func([X, W, matmul])

    ctx = _create_context(
        func,
        target=tvm.target.Target("cuda"),
        rule=schedule_rule.MultiLevelTilingWithIntrin(
            DP4A_INTRIN,
            structure="SSSRRSRS",
            tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
            max_innermost_factor=64,
            vector_load_lens=[1, 2, 3, 4],
            reuse_read=schedule_rule.ReuseType(
                req="must",
                levels=[4],
                scope="shared",
            ),
            reuse_write=schedule_rule.ReuseType(
                req="must",
                levels=[3],
                scope="local",
            ),
        ),
    )

    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)

    expected = [
        """b0 = sch.get_block(name="compute", func_name="main")
sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
l1, l2, l3 = sch.get_loops(block=b0)
l4, l5 = sch.split(loop=l3, factors=[32, 4], preserve_unit_iters=True)
sch.reorder(l5)
b6 = sch.blockize(loop=l5)
sch.annotate(block_or_loop=b6, ann_key="meta_schedule.auto_tensorize", ann_val="dp4a")
l7, l8, l9 = sch.get_loops(block=b6)
v10, v11, v12, v13, v14 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64)
l15, l16, l17, l18, l19 = sch.split(loop=l7, factors=[v10, v11, v12, v13, v14], preserve_unit_iters=True)
v20, v21, v22, v23, v24 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64)
l25, l26, l27, l28, l29 = sch.split(loop=l8, factors=[v20, v21, v22, v23, v24], preserve_unit_iters=True)
v30, v31, v32 = sch.sample_perfect_tile(loop=l9, n=3, max_innermost_factor=64)
l33, l34, l35 = sch.split(loop=l9, factors=[v30, v31, v32], preserve_unit_iters=True)
sch.reorder(l15, l25, l16, l26, l17, l27, l33, l34, l18, l28, l35, l19, l29)
l36 = sch.fuse(l15, l25, preserve_unit_iters=True)
sch.bind(loop=l36, thread_axis="blockIdx.x")
l37 = sch.fuse(l16, l26, preserve_unit_iters=True)
sch.bind(loop=l37, thread_axis="vthread.x")
l38 = sch.fuse(l17, l27, preserve_unit_iters=True)
sch.bind(loop=l38, thread_axis="threadIdx.x")
b39 = sch.cache_write(block=b6, write_buffer_index=0, storage_scope="local")
sch.reverse_compute_at(block=b39, loop=l38, preserve_unit_loops=True)
b40 = sch.cache_read(block=b6, read_buffer_index=0, storage_scope="shared")
sch.compute_at(block=b40, loop=l33, preserve_unit_loops=True)
l41, l42, l43, l44, l45, l46 = sch.get_loops(block=b40)
l47 = sch.fuse(l45, l46, preserve_unit_iters=True)
v48 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])
sch.annotate(block_or_loop=b40, ann_key="meta_schedule.cooperative_fetch", ann_val=v48)
b49 = sch.cache_read(block=b6, read_buffer_index=1, storage_scope="shared")
sch.compute_at(block=b49, loop=l33, preserve_unit_loops=True)
l50, l51, l52, l53, l54, l55 = sch.get_loops(block=b49)
l56 = sch.fuse(l54, l55, preserve_unit_iters=True)
v57 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])
sch.annotate(block_or_loop=b49, ann_key="meta_schedule.cooperative_fetch", ann_val=v57)""".split(
            "\n"
        )
    ]

    check_trace(spaces, expected)


def test_cuda_tensor_core_conv2d():
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.conv2d_nhwc_f16(
                N=1, H=16, W=16, CI=16, CO=16, kernel_size=3, stride=1, padding=1
            )
        ),
        target,
        multi_level_tiling_tensor_core(target=target, scope="shared"),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1

    expected = []
    print("".join(spaces[0].trace.as_python()))
    check_trace(spaces, expected)


def test_cuda_tensor_core_matmul_relu():
    m = n = k = 128
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_relu_fp16(
                n=n,
                m=m,
                k=k,
            )
        ),
        target=target,
        rule=multi_level_tiling_tensor_core(target=target, scope="shared"),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1

    expected = []
    print("".join(spaces[0].trace.as_python()))
    check_trace(spaces, expected)


def test_cuda_tensor_core_matmul_relu_global():
    m = n = k = 128
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_relu_fp16(
                n=n,
                m=m,
                k=k,
            ),
        ),
        target=target,
        rule=[multi_level_tiling_tensor_core(target=target, scope="global"), auto_inline(target)],
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1

    expected = [
        """b0 = sch.get_block(name="C", func_name="main")
sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
b1 = sch.reindex(block=b0, buffer=("write", 0))
b2 = sch.reindex(block=b0, buffer=("read", 0))
b3 = sch.reindex(block=b0, buffer=("read", 1))
sch.transform_layout(block=b0, buffer=("write", 0), index_map=lambda i, j: (i, j, ))
sch.transform_layout(block=b0, buffer=("read", 1), index_map=lambda j, k: (k, j, ))
sch.transform_layout(block=b0, buffer=("read", 0), index_map=lambda i, k: (i, k, ))
sch.transform_block_layout(block=b1, index_map=lambda i, j, k: (i, j, k, ))
sch.transform_block_layout(block=b2, index_map=lambda i, j, k: (i, j, k, ))
sch.transform_block_layout(block=b3, index_map=lambda i, j, k: (i, j, k, ))
sch.transform_block_layout(block=b0, index_map=lambda i, j, k: (i, j, k, ))
l4, l5, l6 = sch.get_loops(block=b0)
l7, l8 = sch.split(loop=l6, factors=[None, 16], preserve_unit_iters=True)
l9, l10 = sch.split(loop=l5, factors=[None, 16], preserve_unit_iters=True)
l11, l12 = sch.split(loop=l4, factors=[None, 16], preserve_unit_iters=True)
l13, l14, l15, l16, l17, l18 = sch.get_loops(block=b0)
sch.reorder(l15, l17, l12, l10, l8)
b19 = sch.blockize(loop=l12)
sch.annotate(block_or_loop=b19, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f32")
sch.annotate(block_or_loop=b19, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_16x16x16_f32")
sch.annotate(block_or_loop=b19, ann_key="warp_execution", ann_val=1)
l20, l21, l22 = sch.get_loops(block=b19)
v23, v24, v25, v26, v27 = sch.sample_perfect_tile(loop=l20, n=5, max_innermost_factor=64)
l28, l29, l30, l31, l32 = sch.split(loop=l20, factors=[v23, v24, v25, v26, v27], preserve_unit_iters=True)
v33, v34, v35, v36, v37 = sch.sample_perfect_tile(loop=l21, n=5, max_innermost_factor=64)
l38, l39, l40, l41, l42 = sch.split(loop=l21, factors=[v33, v34, v35, v36, v37], preserve_unit_iters=True)
v43, v44, v45 = sch.sample_perfect_tile(loop=l22, n=3, max_innermost_factor=64)
l46, l47, l48 = sch.split(loop=l22, factors=[v43, v44, v45], preserve_unit_iters=True)
sch.reorder(l28, l38, l29, l39, l30, l40, l46, l47, l31, l41, l48, l32, l42)
l49 = sch.fuse(l28, l38, preserve_unit_iters=True)
sch.bind(loop=l49, thread_axis="blockIdx.x")
l50 = sch.fuse(l29, l39, preserve_unit_iters=True)
sch.bind(loop=l50, thread_axis="vthread.x")
l51 = sch.fuse(l30, l40, preserve_unit_iters=True)
sch.bind(loop=l51, thread_axis="threadIdx.x")
b52 = sch.cache_write(block=b19, write_buffer_index=0, storage_scope="wmma.accumulator")
sch.reverse_compute_at(block=b52, loop=l51, preserve_unit_loops=True)
b53, = sch.get_consumers(block=b52)
sch.reverse_compute_inline(block=b53)
l54, l55, l56, l57, l58 = sch.get_loops(block=b52)
l59, l60 = sch.split(loop=l58, factors=[None, 16], preserve_unit_iters=True)
l61, l62 = sch.split(loop=l57, factors=[None, 16], preserve_unit_iters=True)
l63, l64, l65, l66, l67, l68, l69 = sch.get_loops(block=b52)
sch.reorder(l68, l62, l60)
sch.tensorize(block_or_loop=l62, tensor_intrin="wmma_store_16x16x16_f32_global")
b70 = sch.cache_read(block=b19, read_buffer_index=0, storage_scope="shared")
sch.compute_at(block=b70, loop=l46, preserve_unit_loops=True)
l71, l72, l73, l74, l75, l76 = sch.get_loops(block=b70)
l77 = sch.fuse(l75, l76, preserve_unit_iters=True)
v78 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])
sch.annotate(block_or_loop=b70, ann_key="meta_schedule.cooperative_fetch", ann_val=v78)
b79 = sch.cache_read(block=b19, read_buffer_index=1, storage_scope="shared")
sch.compute_at(block=b79, loop=l46, preserve_unit_loops=True)
l80, l81, l82, l83, l84, l85 = sch.get_loops(block=b79)
l86 = sch.fuse(l84, l85, preserve_unit_iters=True)
v87 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])
sch.annotate(block_or_loop=b79, ann_key="meta_schedule.cooperative_fetch", ann_val=v87)
b88 = sch.cache_read(block=b19, read_buffer_index=0, storage_scope="wmma.matrix_a")
sch.compute_at(block=b88, loop=l47, preserve_unit_loops=True)
l89, l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b88)
l96, l97 = sch.split(loop=l95, factors=[None, 16], preserve_unit_iters=True)
l98, l99 = sch.split(loop=l94, factors=[None, 16], preserve_unit_iters=True)
l100, l101, l102, l103, l104, l105, l106, l107, l108 = sch.get_loops(block=b88)
sch.reorder(l107, l99, l97)
sch.tensorize(block_or_loop=l99, tensor_intrin="wmma_load_16x16x16_f16_a")
b109 = sch.cache_read(block=b19, read_buffer_index=1, storage_scope="wmma.matrix_b")
sch.compute_at(block=b109, loop=l47, preserve_unit_loops=True)
l110, l111, l112, l113, l114, l115, l116 = sch.get_loops(block=b109)
l117, l118 = sch.split(loop=l116, factors=[None, 16], preserve_unit_iters=True)
l119, l120 = sch.split(loop=l115, factors=[None, 16], preserve_unit_iters=True)
l121, l122, l123, l124, l125, l126, l127, l128, l129 = sch.get_loops(block=b109)
sch.reorder(l128, l120, l118)
sch.tensorize(block_or_loop=l120, tensor_intrin="wmma_load_16x16x16_f16_b")
sch.compute_inline(block=b2)
sch.compute_inline(block=b3)""".split(
            "\n"
        )
    ]
    check_trace(spaces, expected)
    print(spaces[0].mod.script())


def test_multi_level_tiling_non_tensorizable():
    # expected to do nothing on non-tensorizable workloads
    m = n = k = 128
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            # dtype doesn't match tensor intrin
            te_workload.matmul_relu(
                n=n,
                m=m,
                k=k,
            )
        ),
        target=target,
        rule=multi_level_tiling_tensor_core(target=target, scope="global"),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1

    expected = [
        "",
    ]
    print("".join(spaces[0].trace.as_python()))
    check_trace(spaces, expected)


if __name__ == "__main__":
    tvm.testing.main()
