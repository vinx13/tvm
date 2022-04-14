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

from tvm import te, topi, tir
from tvm.meta_schedule.space_generator.post_order_apply import PostOrderApply
from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.schedule_rule import (
    multi_level_tiling,
    multi_level_tiling_tensor_core,
)
from tvm.meta_schedule.testing.space_generation import check_trace
from tvm.meta_schedule.tune_context import TuneContext
from tvm.script import tir as T
from tvm.te import create_prim_func
from tvm.target import Target
from tvm.meta_schedule.testing import tir_tensor_intrin


def _create_context(mod, target, rule) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=[rule],
        task_name="test",
    )
    ctx.space_generator.initialize_with_tune_context(ctx)
    for sch_rule in ctx.sch_rules:
        sch_rule.initialize_with_tune_context(ctx)
    return ctx


def test_cpu_matmul():
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7])",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15])",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21])",
            "sch.reorder(l8, l16, l9, l17, l22, l10, l18, l23, l11, l19)",
            'b24 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")',
            "sch.reverse_compute_at(block=b24, loop=l17, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7])",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15])",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21])",
            "sch.reorder(l8, l16, l9, l17, l22, l10, l18, l23, l11, l19)",
            'b24 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")',
            "sch.reverse_compute_at(block=b24, loop=l16, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7])",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15])",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21])",
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
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7])",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15])",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21])",
            "sch.reorder(l8, l16, l9, l17, l22, l10, l18, l23, l11, l19)",
            "b24, = sch.get_consumers(block=b0)",
            "sch.reverse_compute_at(block=b24, loop=l17, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7])",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15])",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21])",
            "sch.reorder(l8, l16, l9, l17, l22, l10, l18, l23, l11, l19)",
            "b24, = sch.get_consumers(block=b0)",
            "sch.reverse_compute_at(block=b24, loop=l16, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7])",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15])",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21])",
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
            'b1 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")',
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "v5, v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64)",
            "l10, l11, l12, l13, l14 = sch.split(loop=l2, factors=[v5, v6, v7, v8, v9])",
            "v15, v16, v17, v18, v19 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64)",
            "l20, l21, l22, l23, l24 = sch.split(loop=l3, factors=[v15, v16, v17, v18, v19])",
            "v25, v26, v27 = sch.sample_perfect_tile(loop=l4, n=3, max_innermost_factor=64)",
            "l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27])",
            "sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)",
            "l31 = sch.fuse(l10, l20)",
            'sch.bind(loop=l31, thread_axis="blockIdx.x")',
            "l32 = sch.fuse(l11, l21)",
            'sch.bind(loop=l32, thread_axis="vthread.x")',
            "l33 = sch.fuse(l12, l22)",
            'sch.bind(loop=l33, thread_axis="threadIdx.x")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)',
            'b33 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")',
            "sch.reverse_compute_at(block=b33, loop=l32, preserve_unit_loops=True)",
            'b34 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared")',
            "sch.compute_at(block=b34, loop=l27, preserve_unit_loops=True)",
            "l35, l36, l37, l38, l39, l40 = sch.get_loops(block=b34)",
            "l41 = sch.fuse(l39, l40)",
            "v42 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v42)',
            'b43 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b43, loop=l27, preserve_unit_loops=True)",
            "l44, l45, l46, l47, l48, l49 = sch.get_loops(block=b43)",
            "l50 = sch.fuse(l48, l49)",
            "v51 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b43, ann_key="meta_schedule.cooperative_fetch", ann_val=v51)',
            "sch.reverse_compute_at(block=b1, loop=l33, preserve_unit_loops=True)",
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
            'b1 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")',
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "v5, v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64)",
            "l10, l11, l12, l13, l14 = sch.split(loop=l2, factors=[v5, v6, v7, v8, v9])",
            "v15, v16, v17, v18, v19 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64)",
            "l20, l21, l22, l23, l24 = sch.split(loop=l3, factors=[v15, v16, v17, v18, v19])",
            "v25, v26, v27 = sch.sample_perfect_tile(loop=l4, n=3, max_innermost_factor=64)",
            "l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27])",
            "sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)",
            "l31 = sch.fuse(l10, l20)",
            'sch.bind(loop=l31, thread_axis="blockIdx.x")',
            "l32 = sch.fuse(l11, l21)",
            'sch.bind(loop=l32, thread_axis="threadIdx.x")',
            'b33 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")',
            "sch.reverse_compute_at(block=b33, loop=l32, preserve_unit_loops=True)",
            'b34 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared")',
            "sch.compute_at(block=b34, loop=l27, preserve_unit_loops=True)",
            "l35, l36, l37, l38, l39, l40 = sch.get_loops(block=b34)",
            "l41 = sch.fuse(l39, l40)",
            "v42 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v42)',
            'b43 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b43, loop=l27, preserve_unit_loops=True)",
            "l44, l45, l46, l47, l48, l49 = sch.get_loops(block=b43)",
            "l50 = sch.fuse(l48, l49)",
            "v51 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b43, ann_key="meta_schedule.cooperative_fetch", ann_val=v51)',
            "sch.reverse_compute_at(block=b1, loop=l33, preserve_unit_loops=True)",
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


def test_cuda_tensor_core_matmul():
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "l4, l5 = sch.split(loop=l1, factors=[32, 16])",
            "l6, l7 = sch.split(loop=l2, factors=[32, 16])",
            "l8, l9 = sch.split(loop=l3, factors=[32, 16])",
            "l10, l11, l12, l13, l14, l15 = sch.get_loops(block=b0)",
            "sch.reorder(l12, l14, l5, l7, l9)",
            "b16 = sch.blockize(loop=l5)",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync")',
            'sch.annotate(block_or_loop=b16, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill")',
            'b17 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b17, ann_key="meta_schedule.tensor_core_enabled", ann_val="1")',
            'b18 = sch.cache_write(block=b16, write_buffer_index=0, storage_scope="local")',
            'b19 = sch.cache_write(block=b16, write_buffer_index=0, storage_scope="wmma.accumulator")',
            'sch.annotate(block_or_loop=b19, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store")',
            "l20, l21, l22 = sch.get_loops(block=b16)",
            "v23, v24, v25, v26, v27 = sch.sample_perfect_tile(loop=l20, n=5, max_innermost_factor=64)",
            "l28, l29, l30, l31, l32 = sch.split(loop=l20, factors=[v23, v24, v25, v26, v27])",
            "v33, v34, v35, v36, v37 = sch.sample_perfect_tile(loop=l21, n=5, max_innermost_factor=64)",
            "l38, l39, l40, l41, l42 = sch.split(loop=l21, factors=[v33, v34, v35, v36, v37])",
            "v43, v44, v45 = sch.sample_perfect_tile(loop=l22, n=3, max_innermost_factor=64)",
            "l46, l47, l48 = sch.split(loop=l22, factors=[v43, v44, v45])",
            "sch.reorder(l28, l38, l29, l39, l30, l40, l46, l47, l31, l41, l48, l32, l42)",
            "l49 = sch.fuse(l28, l38)",
            'sch.bind(loop=l49, thread_axis="blockIdx.x")',
            "l50 = sch.fuse(l29, l39)",
            'sch.bind(loop=l50, thread_axis="blockIdx.y")',
            "l51 = sch.fuse(l30, l40)",
            'sch.bind(loop=l51, thread_axis="threadIdx.y")',
            'b52 = sch.cache_read(block=b16, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b52, loop=l46, preserve_unit_loops=True)",
            "l53, l54, l55, l56, l57, l58 = sch.get_loops(block=b52)",
            "l59 = sch.fuse(l57, l58)",
            "v60 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b52, ann_key="meta_schedule.cooperative_fetch", ann_val=v60)',
            'b61 = sch.cache_read(block=b16, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b61, loop=l46, preserve_unit_loops=True)",
            "l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b61)",
            "l68 = sch.fuse(l66, l67)",
            "v69 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b61, ann_key="meta_schedule.cooperative_fetch", ann_val=v69)',
            'b70 = sch.cache_read(block=b16, read_buffer_index=1, storage_scope="wmma.matrix_a")',
            'b71 = sch.cache_read(block=b16, read_buffer_index=2, storage_scope="wmma.matrix_b")',
            "sch.compute_at(block=b70, loop=l48, preserve_unit_loops=True)",
            "sch.compute_at(block=b71, loop=l48, preserve_unit_loops=True)",
            'sch.annotate(block_or_loop=b70, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_a")',
            'sch.annotate(block_or_loop=b71, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_b")',
            "sch.reverse_compute_at(block=b19, loop=l51, preserve_unit_loops=True)",
            "sch.reverse_compute_at(block=b18, loop=l51, preserve_unit_loops=True)",
        ]
    ]
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_fp16(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling_tensor_core(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    print(spaces[0].mod.script())
    # check_trace(spaces, expected)


def test_cuda_tensor_core_matmul_relu():
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "l4, l5 = sch.split(loop=l1, factors=[32, 16])",
            "l6, l7 = sch.split(loop=l2, factors=[32, 16])",
            "l8, l9 = sch.split(loop=l3, factors=[32, 16])",
            "l10, l11, l12, l13, l14, l15 = sch.get_loops(block=b0)",
            "sch.reorder(l12, l14, l5, l7, l9)",
            "b16 = sch.blockize(loop=l5)",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync")',
            'sch.annotate(block_or_loop=b16, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill")',
            'b17 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b17, ann_key="meta_schedule.tensor_core_enabled", ann_val="1")',
            'b18 = sch.cache_write(block=b16, write_buffer_index=0, storage_scope="local")',
            'b19 = sch.cache_write(block=b16, write_buffer_index=0, storage_scope="wmma.accumulator")',
            'sch.annotate(block_or_loop=b19, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store")',
            "l20, l21, l22 = sch.get_loops(block=b16)",
            "v23, v24, v25, v26, v27 = sch.sample_perfect_tile(loop=l20, n=5, max_innermost_factor=64)",
            "l28, l29, l30, l31, l32 = sch.split(loop=l20, factors=[v23, v24, v25, v26, v27])",
            "v33, v34, v35, v36, v37 = sch.sample_perfect_tile(loop=l21, n=5, max_innermost_factor=64)",
            "l38, l39, l40, l41, l42 = sch.split(loop=l21, factors=[v33, v34, v35, v36, v37])",
            "v43, v44, v45 = sch.sample_perfect_tile(loop=l22, n=3, max_innermost_factor=64)",
            "l46, l47, l48 = sch.split(loop=l22, factors=[v43, v44, v45])",
            "sch.reorder(l28, l38, l29, l39, l30, l40, l46, l47, l31, l41, l48, l32, l42)",
            "l49 = sch.fuse(l28, l38)",
            'sch.bind(loop=l49, thread_axis="blockIdx.x")',
            "l50 = sch.fuse(l29, l39)",
            'sch.bind(loop=l50, thread_axis="blockIdx.y")',
            "l51 = sch.fuse(l30, l40)",
            'sch.bind(loop=l51, thread_axis="threadIdx.y")',
            'b52 = sch.cache_read(block=b16, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b52, loop=l46, preserve_unit_loops=True)",
            "l53, l54, l55, l56, l57, l58 = sch.get_loops(block=b52)",
            "l59 = sch.fuse(l57, l58)",
            "v60 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b52, ann_key="meta_schedule.cooperative_fetch", ann_val=v60)',
            'b61 = sch.cache_read(block=b16, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b61, loop=l46, preserve_unit_loops=True)",
            "l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b61)",
            "l68 = sch.fuse(l66, l67)",
            "v69 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b61, ann_key="meta_schedule.cooperative_fetch", ann_val=v69)',
            'b70 = sch.cache_read(block=b16, read_buffer_index=1, storage_scope="wmma.matrix_a")',
            'b71 = sch.cache_read(block=b16, read_buffer_index=2, storage_scope="wmma.matrix_b")',
            "sch.compute_at(block=b70, loop=l48, preserve_unit_loops=True)",
            "sch.compute_at(block=b71, loop=l48, preserve_unit_loops=True)",
            'sch.annotate(block_or_loop=b70, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_a")',
            'sch.annotate(block_or_loop=b71, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_b")',
            "sch.reverse_compute_at(block=b19, loop=l51, preserve_unit_loops=True)",
            "sch.reverse_compute_at(block=b18, loop=l51, preserve_unit_loops=True)",
        ]
    ]
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_relu_fp16(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling_tensor_core(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    print(spaces[0].mod.script())
    # check_trace(spaces, expected)


def conv2d_nhwc_fp16(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    inputs = te.placeholder((N, H, W, CI), name="inputs", dtype="float16")
    weight = te.placeholder((kernel_size, kernel_size, CI // groups, CO), name="weight", dtype="float16")
    batch_size, in_h, in_w, _ = inputs.shape
    k_h, k_w, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups

    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    rc = te.reduce_axis((0, channel_per_group), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, 0])
    output = te.compute(
        (batch_size, out_h, out_w, out_channel),
        lambda n, h, w, co: te.sum(
            (
                tir.Cast(value=padded[
                    n,
                    h * stride + rh * dilation,
                    w * stride + rw * dilation,
                    co // out_channel_per_group * channel_per_group + rc,
                ], dtype="float32")
                * tir.Cast(value=weight[rh, rw, rc, co], dtype="float32")
            ),
            axis=[rh, rw, rc],
        ),
        name="conv2d_nhwc",
    )
    return (inputs, weight, output)


def test_cuda_tensor_core_conv2d():
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
           conv2d_nhwc_fp16(N=32, H=16, W=16, CI=64, CO=64, kernel_size=3, padding=1, dilation=1)
        ),
        target=target,
        rule=multi_level_tiling_tensor_core(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    print(spaces[0].mod.script())


if __name__ == "__main__":
    test_cpu_matmul()
    test_cpu_matmul_relu()
    test_cuda_matmul()
    test_cuda_matmul_relu()
    test_cuda_sum_with_trivial_block_iter()
    test_cuda_tensor_core_matmul()
    test_cuda_tensor_core_matmul_relu()
    test_cuda_tensor_core_conv2d()
