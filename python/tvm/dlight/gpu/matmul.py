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
# pylint: disable=missing-docstring, invalid-name
"""A GEMM schedule rule for GPU operators."""
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Set, Tuple

from tvm import DataType, tir
from tvm.ir import Range
from tvm.target import Target
from tvm.tir import IterVar, PrimExpr, Var
from tvm.tir.analysis import undefined_vars
from tvm.tir.schedule.schedule import BlockRV
from tvm.tir.stmt import ForKind
from tvm.tir.tensor_intrin.cuda import (
    LDMATRIX_16x16_A_DYN_INTRIN,
    LDMATRIX_16x16_B_DYN_INTRIN,
    LDMATRIX_16x16_B_TRANS_DYN_INTRIN,
    MMA_f16f16f32_TRANS_INTRIN,
    MMA_fill_16x16_f32_INTRIN,
    MMA_store_16x16_f32_global_INTRIN,
    MMA_store_16x16_f32_shared_dyn_INTRIN,
    shared_16x16_to_ldmatrix_32x8_layout,
)

from . import utils
from ..base import ScheduleRule, analysis


def _collect_producers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for producer in sch.get_producers(block):
        result.append(producer)
        result.extend(_collect_producers(sch, producer))
    return result


def _collect_consumers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for consumer in sch.get_consumers(block):
        result.append(consumer)
        result.extend(_collect_consumers(sch, consumer))
    return result


def auto_inline_producers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        producers = _collect_producers(sch, block)
        for producer in producers:
            try:
                sch.compute_inline(producer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        consumers = _collect_consumers(sch, block)
        for consumer in consumers:
            try:
                sch.compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        for consumer in consumers:
            try:
                sch.reverse_compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


class IterKind(Enum):
    """Iter kinds for GEMM-liked programs.
    We can simplify the computation to C[S, I, J] += A[S, I, K] * B[S, J, K],
    where `I, J, K` are fundamental axes for gemm and `S` represents all
    other spatial axes (e.g. batches)
    kIter_S: spatial axes
    kIter_I: I axes
    kIter_J: J axes
    kIter_K: K axes
    kIter_T: trivial axes (i.e. with extent 1)
    """

    kIter_S = 0
    kIter_I = 1
    kIter_J = 2
    kIter_K = 3
    kIter_T = 4


@dataclass
class IterTrait:
    kind: IterKind
    extent: PrimExpr


def _is_one(x: PrimExpr) -> bool:
    return isinstance(x, tir.IntImm) and x.value == 1


def make_iter_fusion_index_map(
    traits: List[IterTrait],
    kind_order: List[IterKind],
) -> tir.IndexMap:
    fused_iters: Dict[IterKind, PrimExpr] = {}
    input_iters: List[tir.Var] = []
    for i, trait in enumerate(traits):
        v_i = tir.Var(f"i{i}", trait.extent.dtype)
        input_iters.append(v_i)
        if trait.kind == IterKind.kIter_T:
            continue
        if trait.kind not in kind_order:
            raise ValueError(f"Unknown iter kind {trait.kind}")
        if trait.kind in fused_iters:
            fused_iters[trait.kind] = fused_iters[trait.kind] * trait.extent + v_i
        else:
            fused_iters[trait.kind] = v_i

    final_indices: List[tir.PrimExpr] = [
        fused_iters.get(kind, tir.IntImm(traits[0].extent.dtype, 0)) for kind in kind_order
    ]

    return tir.IndexMap(input_iters, final_indices, None)


def detect_iter_traits(block: tir.Block) -> Optional[Tuple[List[IterTrait]]]:
    """Detect iter traits based on the pattern C[S, I, J] += A[S, I, K] * B[S, J, K]

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    Returns
    -------
    traits : Optional[Tuple[List[IterTrait]]]
        The detected iter traits for axes in A, B and C. None if the block
        does not match the pattern.

    """

    if len(block.reads) != 2 or len(block.writes) != 1:
        return None

    def get_access_axes(region: List[Range]) -> Set[Var]:
        axes: Set[Var] = set()
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            axes = axes.union(set(undefined_vars(r.min)))
        return axes

    try:
        A_axes = get_access_axes(block.reads[0].region)
        B_axes = get_access_axes(block.reads[1].region)
        C_axes = get_access_axes(block.writes[0].region)
    except ValueError:
        return None

    traits: Dict[Var, IterTrait] = {}
    for iter_var in block.iter_vars:
        var = iter_var.var
        kind: IterKind
        if _is_one(iter_var.dom.extent):
            kind = IterKind.kIter_T
        elif iter_var.iter_type == iter_var.DataPar:
            if (var in A_axes and var in B_axes and var in C_axes) or (
                # False
                isinstance(iter_var.dom.extent, tir.Var)
                and iter_var.dom.extent.name == "b"
            ):
                kind = IterKind.kIter_S
            elif var in A_axes and var in C_axes:
                kind = IterKind.kIter_I
            elif var in B_axes and var in C_axes:
                kind = IterKind.kIter_J
            else:
                return None
        elif iter_var.iter_type == tir.IterVar.CommReduce:
            if var in A_axes and var in B_axes and var not in C_axes:
                kind = IterKind.kIter_K
            else:
                return None
        else:
            return None
        traits[var] = IterTrait(kind, iter_var.dom.extent)

    # A Gemm-kernel requires have I, J and K axes
    gemm_traits = {IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K}
    if {x.kind for x in traits.values()}.intersection(gemm_traits) != gemm_traits:
        return None

    A_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in A_axes]
    B_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in B_axes]
    C_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in C_axes]
    block_traits = [traits[i.var] for i in block.iter_vars]
    return A_traits, B_traits, C_traits, block_traits


def get_index_map(block: tir.Block) -> Optional[Tuple[tir.IndexMap, ...]]:
    """Get index maps for the block

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    Returns
    -------
    index_maps : Optional[Tuple[tir.IndexMap]]
        The index maps for the block, or None if the block is not a gemm-liked kernel
    """
    traits = detect_iter_traits(block)
    if traits is None:
        return None
    A_traits, B_traits, C_traits, block_traits = traits

    A_index_map = make_iter_fusion_index_map(
        A_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_K]
    )
    B_index_map = make_iter_fusion_index_map(
        B_traits, [IterKind.kIter_S, IterKind.kIter_J, IterKind.kIter_K]
    )
    C_index_map = make_iter_fusion_index_map(
        C_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J]
    )
    matmul_index_map = make_iter_fusion_index_map(
        block_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K]
    )

    return (
        matmul_index_map,
        A_index_map,
        B_index_map,
        C_index_map,
    )


def get_reduction_blocks(sch, blocks) -> Optional[List[BlockRV]]:
    # Get the main computation block
    def is_reduction(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.CommReduce, IterVar.DataPar}

    def is_spatial(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) != 1:
        return None

    return reduction_blocks


def check_sm_version(arch: str) -> int:
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1


# class MatmulTensorizationMMA(ScheduleRule):
#     """
#     The schedule rule for float16 tensor core matmul computation.
#     func with attr 'dlight.do_not_tensorize' will not be tensorized.
#     """

#     def apply(  # pylint: disable=too-many-locals,missing-docstring
#         self,
#         func: tir.PrimFunc,
#         target: Target,
#         _: bool,
#     ) -> Optional[tir.Schedule]:
#         sch = tir.Schedule(func)
#         root_block = analysis.get_root_block(sch)
#         blocks = sch.get_child_blocks(root_block)

#         if func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys():
#             return None

#         reduction_blocks = get_reduction_blocks(sch, blocks)
#         if reduction_blocks is None:
#             return None

#         main_block = reduction_blocks[0]
#         block_stmt = sch.get(main_block)
#         index_maps = get_index_map(block_stmt)
#         if index_maps is None:
#             return None
#         matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

#         # Start Schedule
#         # Step 0. Get schedule config.
#         # NOTE: we can analyze the config by the hardware spec in the future

#         block_m = 128
#         block_n = 128
#         block_k = 32

#         # tensor core intrinsic size
#         micro_size_m = 16
#         micro_size_n = 16
#         micro_size_k = 16

#         thread_z = 2
#         thread_y = 2
#         warp_size = 32
#         thread_cnt = thread_y * thread_z * warp_size

#         vector_size = 8
#         unroll_depth = 256

#         # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
#         # block = sch.reindex(main_block, ("read", 0))
#         # sch.transform_layout(block, ("write", 0), a_index_map)
#         # block = sch.reindex(main_block, ("read", 1))
#         # sch.transform_layout(block, ("write", 0), b_index_map)
#         # block = sch.reindex(main_block, ("write", 0))
#         # sch.transform_layout(block, ("read", 0), c_index_map)
#         # sch.transform_block_layout(main_block, matmul_index_map)

#         # batch, i, j, k = sch.get_loops(main_block)
#         # main_block_stmt = sch.get(main_block)
#         # buffer_regions = list(main_block_stmt.reads) + list(main_block_stmt.writes)

#         # # Supported data types:
#         # # fp16, fp16, fp16: fp16 precision
#         # # fp16, fp16, fp32: fp16 mixed precision
#         # dtype_a, dtype_b, dtype_c = [DataType(region.buffer.dtype) for region in buffer_regions]
#         # input_b, input_m, input_n, input_k = [sch.get(loop).extent for loop in [batch, i, j, k]]
#         # l2_size = target.l2_cache_size_bytes
#         # dtype_a_bytes, dtype_b_bytes, dtype_c_bytes = [
#         #     math.ceil(d.bits / 8) for d in [dtype_a, dtype_b, dtype_c]
#         # ]
#         # bank_size_bytes = 32 * 4
#         # bank_cnt_a, bank_cnt_b = bank_size_bytes // dtype_a_bytes, bank_size_bytes // dtype_b_bytes

#         # def get_z_order_factor(l2_size, input_k, dtype_bytes, input_spatial, block_size):
#         #     if l2_size != 0 and isinstance(input_k, (int, tir.IntImm)):
#         #         z_order_factor = l2_size / 3 / int(input_k) / dtype_bytes / block_size
#         #         if isinstance(input_spatial, (int, tir.IntImm)):
#         #             block_cnt = math.ceil(int(input_spatial) / block_size)
#         #             z_order_factor = math.ceil(block_cnt / math.ceil(block_cnt / z_order_factor))
#         #         else:
#         #             z_order_factor = math.floor(z_order_factor)
#         #         return [None, z_order_factor]
#         #     else:
#         #         return [4, None]

#         # z_order_factor_m = get_z_order_factor(l2_size, input_k, dtype_a_bytes, input_m, block_m)
#         # z_order_factor_n = get_z_order_factor(l2_size, input_k, dtype_b_bytes, input_n, block_n)

#         # z_order_factor_m = [1, None]
#         # z_order_factor_n = [1, None]

#         # print(f"z_order_factor_m={z_order_factor_m}, z_order_factor_n={z_order_factor_n}")

#         # # Step 2. Padding for dynamic shape kernels
#         # sch.pad_einsum(
#         #     main_block,
#         #     [
#         #         1,
#         #         (z_order_factor_m[0] or z_order_factor_m[1]) * block_m,
#         #         (z_order_factor_n[0] or z_order_factor_n[1]) * block_n,
#         #         block_k,
#         #     ],
#         # )

#         sch.compute_inline(sch.get_block("T_transpose"))

#         i_factors, j_factors, k_factors = [4, 8, 2, 4, 1], [1, 64, 2, 1, 2], [128, 2, 1]

#         b, i, j, k = sch.get_loops(main_block)
#         i, i_tc = sch.split(i, factors=[None, micro_size_m])
#         j, j_tc = sch.split(j, factors=[None, micro_size_n])
#         k, k_tc = sch.split(k, factors=[None, micro_size_k])

#         sch.reorder(i, j, k, i_tc, j_tc, k_tc)

#         block_outer = sch.blockize(i_tc)
#         block_inner = main_block

#         num_ty = i_factors[2] * j_factors[2]

#         i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
#         j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
#         k0, k1, k2 = sch.split(k, k_factors)

#         sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3, k2, i4, j4)

#         block_idy = sch.fuse(b, i0, j0)
#         block_idx = sch.fuse(i1, j1)
#         thread_idy = sch.fuse(j2, i2)
#         sch.bind(block_idy, "blockIdx.y")
#         sch.bind(block_idx, "blockIdx.x")
#         sch.bind(thread_idy, "threadIdx.y")

#         def fetch_to_shared(block, idx, ndim):
#             block_read = sch.cache_read(block, idx, "shared.dyn")
#             sch.compute_at(block_read, k0)
#             vector_size = 8
#             warp_size = 32
#             fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
#             _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])
#             sch.bind(f_2, "threadIdx.x")
#             sch.bind(f_1, "threadIdx.y")
#             sch.vectorize(f_3)
#             offset = 8
#             sch.storage_align(block_read, 0, axis=-2, factor=32, offset=offset)
#             auto_inline_producers(sch, block_read)
#             return block_read

#         fetch_to_shared(block_outer, 0, 2)
#         fetch_to_shared(block_outer, 1, 2)

#         A_warp = sch.cache_read(block_outer, 0, "warp")
#         B_warp = sch.cache_read(block_outer, 1, "warp")

#         sch.compute_at(A_warp, k1)
#         sch.compute_at(B_warp, k1)

#         # def store_to_shared(block, idx, ndim):
#         #     block_write = sch.cache_write(block, idx, "shared.dyn")
#         #     sch.reverse_compute_at(block_write, block_idx)
#         #     vector_size = 8
#         #     warp_size = 32
#         #     fused = sch.fuse(*sch.get_loops(block_write)[-ndim:])
#         #     _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])
#         #     sch.bind(f_2, "threadIdx.x")
#         #     sch.bind(f_1, "threadIdx.y")
#         #     sch.vectorize(f_3)
#         #     offset = 8
#         #     sch.storage_align(block_write, 0, axis=-2, factor=32, offset=offset)
#         #     auto_inline_consumers(sch, block_write)
#         #     return block_write

#         # store_to_shared(block_outer, 0, 2)

#         C_warp = sch.cache_write(block_outer, 0, "warp")
#         sch.reverse_compute_at(C_warp, thread_idy)

#         ii, jj = sch.get_loops(C_warp)[-2:]
#         io, ii = sch.split(ii, factors=[None, 16])
#         jo, ji = sch.split(jj, factors=[None, 16])
#         sch.reorder(io, jo, ii, ji)


#         block_init_c = sch.decompose_reduction(block_outer, sch.get_loops(block_outer)[3])
#         block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

#         def tile_wmma_fragment(block_read, height, width):
#             i, j = sch.get_loops(block_read)[-2:]
#             i0, i1 = sch.split(i, factors=[None, height])
#             j0, j1 = sch.split(j, factors=[None, width])
#             sch.reorder(i0, j0, i1, j1)
#             return i1

#         loop_a = tile_wmma_fragment(A_warp, 16, micro_size_k)
#         loop_b = tile_wmma_fragment(B_warp, 16, micro_size_k)
#         def index_map(b, i, j):
#             return (
#                 b,
#                 i // 16,
#                 j // 16,
#                 *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
#             )

#         def index_map1(i, j):
#             return (
#                 i // 16,
#                 j // 16,
#                 *shared_16x16_to_ldmatrix_32x8_layout(i % 16, j % 16),
#             )

#         sch.transform_layout(A_warp, ("write", 0), index_map)
#         sch.transform_layout(B_warp, ("write", 0), index_map1)
#         sch.transform_layout(C_warp, ("read", 0), index_map)

#         sch.tensorize(loop_a, LDMATRIX_16x16_A_DYN_INTRIN)
#         sch.tensorize(loop_b, LDMATRIX_16x16_B_DYN_INTRIN)
#         sch.tensorize(sch.get_loops(block_inner)[-3], MMA_f16f16f32_TRANS_INTRIN)
#         sch.tensorize(sch.get_loops(block_init_c_inner)[-2], MMA_fill_16x16_f32_INTRIN)
#         sch.tensorize(sch.get_loops(C_warp)[-2], MMA_store_16x16_f32_global_INTRIN)

#         return sch


class MatmulTensorizationMMA(ScheduleRule):
    """
    The schedule rule for float16 tensor core matmul computation.
    func with attr 'dlight.do_not_tensorize' will not be tensorized.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        index_maps = get_index_map(block_stmt)
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        block_m = 128
        block_n = 128
        block_k = 32

        # tensor core intrinsic size
        micro_size_m = 16
        micro_size_n = 16
        micro_size_k = 16

        thread_z = 2
        thread_y = 2
        warp_size = 32
        thread_cnt = thread_y * thread_z * warp_size
        k_cnt = block_k // micro_size_k

        vector_size = 8
        unroll_depth = 256

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        batch, i, j, k = sch.get_loops(main_block)
        main_block_stmt = sch.get(main_block)
        buffer_regions = list(main_block_stmt.reads) + list(main_block_stmt.writes)

        # Supported data types:
        # fp16, fp16, fp16: fp16 precision
        # fp16, fp16, fp32: fp16 mixed precision
        dtype_a, dtype_b, dtype_c = [DataType(region.buffer.dtype) for region in buffer_regions]
        input_b, input_m, input_n, input_k = [sch.get(loop).extent for loop in [batch, i, j, k]]
        l2_size = target.l2_cache_size_bytes
        dtype_a_bytes, dtype_b_bytes, dtype_c_bytes = [
            math.ceil(d.bits / 8) for d in [dtype_a, dtype_b, dtype_c]
        ]
        bank_size_bytes = 32 * 4
        bank_cnt_a, bank_cnt_b = bank_size_bytes // dtype_a_bytes, bank_size_bytes // dtype_b_bytes

        def get_z_order_factor(l2_size, input_k, dtype_bytes, input_spatial, block_size):
            if l2_size != 0 and isinstance(input_k, (int, tir.IntImm)):
                z_order_factor = l2_size / 3 / int(input_k) / dtype_bytes / block_size
                if isinstance(input_spatial, (int, tir.IntImm)):
                    block_cnt = math.ceil(int(input_spatial) / block_size)
                    z_order_factor = math.ceil(block_cnt / math.ceil(block_cnt / z_order_factor))
                else:
                    z_order_factor = math.floor(z_order_factor)
                return [None, z_order_factor]
            else:
                return [4, None]

        z_order_factor_m = get_z_order_factor(l2_size, input_k, dtype_a_bytes, input_m, block_m)
        z_order_factor_n = get_z_order_factor(l2_size, input_k, dtype_b_bytes, input_n, block_n)

        z_order_factor_m = [1, None]
        z_order_factor_n = [1, None]
        # z_order_factor_m = [None, 16]
        # z_order_factor_n = [None, 16]

        print(f"z_order_factor_m={z_order_factor_m}, z_order_factor_n={z_order_factor_n}")

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                (z_order_factor_m[0] or z_order_factor_m[1]) * block_m,
                (z_order_factor_n[0] or z_order_factor_n[1]) * block_n,
                block_k,
            ],
        )

        # Step 3. Schedule matmul to use tensor core

        # inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, micro_size_m])
        j, j_inner = sch.split(j, factors=[None, micro_size_n])
        k, k_inner = sch.split(k, factors=[None, micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)

        block_inner = main_block
        block_outer = sch.blockize(i_inner)

        # split factors for i, j, and k
        in_wrap_block_cnt_m = block_m // thread_z // micro_size_m
        in_wrap_block_cnt_n = block_n // thread_y // micro_size_n
        in_wrap_block_cnt_k = block_k // micro_size_k

        i_factors = z_order_factor_m + [thread_z, in_wrap_block_cnt_m]
        j_factors = z_order_factor_n + [thread_y, in_wrap_block_cnt_n]
        k_factors = [None, in_wrap_block_cnt_k]

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, factors=k_factors)

        sch.reorder(i0, j0, i1, j1, k0, i2, j2, k1, i3, j3)
        block_axis = sch.fuse(batch, i0, j0, i1, j1)

        sch.bind(block_axis, "blockIdx.x")
        sch.bind(i2, "threadIdx.z")
        sch.bind(j2, "threadIdx.y")

        def fetch_input(block_outer, read_buffer_idx, tensor_name: Literal["A", "B"]):
            block_read = sch.cache_read(block_outer, read_buffer_idx, "shared.dyn")
            sch.compute_at(block_read, k0)
            fused = sch.fuse(*sch.get_loops(block_read)[-2:])

            f0, f1, f2, f3, f4 = sch.split(
                fused, [None, thread_z, thread_y, warp_size, vector_size]
            )

            if tensor_name == "A":
                micro_size_spatial = micro_size_m
            else:
                micro_size_spatial = micro_size_n

            sch.bind(f1, "threadIdx.z")
            sch.bind(f2, "threadIdx.y")
            sch.bind(f3, "threadIdx.x")
            sch.vectorize(f4)

            sch.annotate(block_read, ann_key="permuted_layout", ann_val=f"g2s_{tensor_name}")

            auto_inline_producers(sch, block_read)

            mma_read = sch.cache_read(block_outer, read_buffer_idx, "warp")
            sch.compute_at(mma_read, k1)

            sch.split(sch.get_loops(mma_read)[-2], [None, micro_size_spatial])

            sch.transform_layout(
                mma_read,
                ("write", 0),
                lambda v0, v1, v2: (
                    v1 // micro_size_spatial,
                    v2 // micro_size_k,
                    *shared_16x16_to_ldmatrix_32x8_layout(
                        v1 % micro_size_spatial, v2 % micro_size_k
                    ),
                ),
            )

            mma_read_block = sch.blockize(sch.get_loops(mma_read)[-2])
            sch.annotate(mma_read_block, ann_key="permuted_layout", ann_val=f"s2l_{tensor_name}")
            return block_read, mma_read

        block_read_a, mma_read_a = fetch_input(block_outer, 0, "A")
        block_read_b, mma_read_b = fetch_input(block_outer, 1, "B")

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        def store_output(block_outer, write_buffer_idx, write_loop_ndim, block_sizes):
            block_write = sch.cache_write(block_outer, write_buffer_idx, "shared.dyn")
            sch.reverse_compute_at(block_write, block_axis)

            fused = sch.fuse(*sch.get_loops(block_write)[-write_loop_ndim:])

            f0, f1, f2, f3, f4 = sch.split(
                fused, [None, thread_z, thread_y, warp_size, vector_size]
            )

            block_m, block_n, micro_size_m, micro_size_n = block_sizes

            sch.bind(f1, "threadIdx.z")
            sch.bind(f2, "threadIdx.y")
            sch.bind(f3, "threadIdx.x")
            sch.vectorize(f4)
            auto_inline_consumers(sch, block_write)

            store = sch.cache_write(block_outer, write_buffer_idx, "warp")
            v0, v1, v2 = sch.get_loops(store)[-3:]
            v11, v12, v13 = sch.split(v1, factors=[thread_z, None, micro_size_m])
            v21, v22, v23 = sch.split(v2, factors=[thread_y, None, micro_size_n])
            sch.reorder(v11, v21, v12, v22, v13, v23)
            sch.bind(v11, "threadIdx.z")
            sch.bind(v21, "threadIdx.y")

            sch.transform_layout(
                store,
                ("read", 0),
                lambda v0, v1, v2: (
                    v1 // micro_size_m,
                    v2 // micro_size_n,
                    *shared_16x16_to_ldmatrix_32x8_layout(v1 % micro_size_m, v2 % micro_size_n),
                ),
            )

            return block_write, store

        block_write, store = store_output(
            block_outer, 0, 2, [block_m, block_n, micro_size_m, micro_size_n]
        )

        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # unroll k
        # sch.annotate(k0, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        # sch.unroll(k0)
        # sch.annotate(k0, ann_key="pragma_unroll_explicit", ann_val=0)
        # k00, k01 = sch.split(k0, factors=[None, 8])
        # sch.annotate(k01, ann_key="pragma_unroll_explicit", ann_val=0)
        # sch.unroll(k01)

        # Tensorization by hardware intrinsics
        # from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
        #     get_wmma_intrin_group,
        # )

        # intrin_group = get_wmma_intrin_group(
        #     load_scope="shared.dyn",
        #     store_scope="shared.dyn",
        #     in_dtype=str(dtype_a),
        #     out_dtype=str(dtype_c),
        #     trans_b=True,
        # )

        sch.tensorize(sch.get_loops(block_init_c_inner)[-2], MMA_fill_16x16_f32_INTRIN)
        sch.tensorize(sch.get_loops(mma_read_a)[-2], LDMATRIX_16x16_A_DYN_INTRIN)
        sch.tensorize(sch.get_loops(mma_read_b)[-2], LDMATRIX_16x16_B_TRANS_DYN_INTRIN)
        sch.tensorize(sch.get_loops(block_inner)[-3], MMA_f16f16f32_TRANS_INTRIN)
        sch.tensorize(sch.get_loops(store)[-2], MMA_store_16x16_f32_shared_dyn_INTRIN)

        # sch.transform_layout(
        #     block_read_a,
        #     ("write", 0),
        #     lambda v0, v1, v2: (
        #         v0,
        #         v1 // block_m,
        #         v2 // block_k,
        #         v1 % block_m // 2,
        #         v1 % 2 * 32 + v2 % block_k,
        #     ),
        # )

        # sch.transform_layout(
        #     block_read_a,
        #     ("write", 0),
        #     lambda v0, v1, v2, v3, v4: (v0, v1, v2, v3, v4 ^ (v3 & 3 << 3)),
        # )

        # sch.transform_layout(
        #     block_read_b,
        #     ("write", 0),
        #     lambda v0, v1, v2: (
        #         v0,
        #         v1 // block_n,
        #         v2 // block_k,
        #         v1 % block_n // 2,
        #         v1 % 2 * 32 + v2 % block_k,
        #     ),
        # )

        # sch.transform_layout(
        #     block_read_b,
        #     ("write", 0),
        #     lambda v0, v1, v2, v3, v4: (v0, v1, v2, v3, v4 ^ (v3 & 3 << 3)),
        # )

        # sch.transform_layout(
        #     block_read_b,
        #     ("write", 0),
        #     lambda v0, v1, v2: (
        #         v1 // block_n,
        #         v2 // block_k,
        #         v1 % block_n,
        #         v2 % block_k,
        #     ),
        # )

        # sch.transform_layout(
        #     block_write,
        #     ("read", 0),
        #     lambda v0, v1, v2: (
        #         v1 // block_m,
        #         v2 // block_n,
        #         v1 % block_m,
        #         v2 % block_n,
        #     ),
        # )

        # sch.transform_layout(
        #     sch.get_block("p_output0_intermediate_reindex_shared.dyn_warp_o"),
        #     ("write", 0),
        #     lambda v0, v1, v2: (
        #         v1 // block_m,
        #         v2 // block_n,
        #         v1 % block_m,
        #         v2 % block_n,
        #     ),
        # )

        # sch.transform_layout(
        #     block_read_a,
        #     ("write", 0),
        #     lambda v0, v1, v2: (
        #         v1 // block_m,
        #         v2 // block_k,
        #         v1 % block_m // 2,
        #         v1 % block_m % 2 * block_k + v2 % block_k,
        #         # (v1 % block_m * block_k + v2 % block_k) // bank_cnt_a,
        #         # (v1 % block_m * block_k + v2 % block_k) % bank_cnt_a,
        #     ),
        # )

        # sch.transform_layout(
        #     block_read_b,
        #     ("write", 0),
        #     lambda v0, v1, v2: (
        #         v1 // block_m,
        #         v2 // block_k,
        #         v1 % block_m // 2,
        #         v1 % block_m % 2 * block_k + v2 % block_k,
        #         # (v1 % block_m * block_k + v2 % block_k) // bank_cnt_a,
        #         # (v1 % block_m * block_k + v2 % block_k) % bank_cnt_a,
        #     ),
        # )

        # sch.transform_layout(
        #     block_write,
        #     ("read", 0),
        #     lambda v0, v1, v2: (
        #         v1 // block_m,
        #         v2 // block_n,
        #         v1 % block_m,
        #         v2 % block_n,
        #     ),
        # )

        # async pipeline
        sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 3])
        sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
        sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0])

        return sch


class MatmulTensorization(ScheduleRule):
    """
    The schedule rule for float16 tensor core matmul computation.
    func with attr 'dlight.do_not_tensorize' will not be tensorized.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if func.attrs is not None and "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        index_maps = get_index_map(block_stmt)
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        block_m = 128
        block_n = 128
        block_k = 32

        # tensor core intrinsic size
        micro_size_m = 16
        micro_size_n = 16
        micro_size_k = 16

        thread_z = 2
        thread_y = 2
        warp_size = 32
        thread_cnt = thread_y * thread_z * warp_size

        vector_size = 8
        unroll_depth = 256

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        batch, i, j, k = sch.get_loops(main_block)
        main_block_stmt = sch.get(main_block)
        buffer_regions = list(main_block_stmt.reads) + list(main_block_stmt.writes)

        # Supported data types:
        # fp16, fp16, fp16: fp16 precision
        # fp16, fp16, fp32: fp16 mixed precision
        dtype_a, dtype_b, dtype_c = [DataType(region.buffer.dtype) for region in buffer_regions]
        input_b, input_m, input_n, input_k = [sch.get(loop).extent for loop in [batch, i, j, k]]
        l2_size = target.l2_cache_size_bytes
        dtype_a_bytes, dtype_b_bytes, dtype_c_bytes = [
            math.ceil(d.bits / 8) for d in [dtype_a, dtype_b, dtype_c]
        ]

        def get_z_order_factor(l2_size, input_k, dtype_bytes, input_spatial, block_size):
            if l2_size != 0 and isinstance(input_k, (int, tir.IntImm)):
                z_order_factor = l2_size / 3 / int(input_k) / dtype_bytes / block_size
                if isinstance(input_spatial, (int, tir.IntImm)):
                    block_cnt = math.ceil(int(input_spatial) / block_size)
                    z_order_factor = math.ceil(block_cnt / math.ceil(block_cnt / z_order_factor))
                else:
                    z_order_factor = math.floor(z_order_factor)
                return [None, z_order_factor]
            else:
                return [4, None]

        z_order_factor_m = get_z_order_factor(l2_size, input_k, dtype_a_bytes, input_m, block_m)
        z_order_factor_n = get_z_order_factor(l2_size, input_k, dtype_b_bytes, input_n, block_n)

        z_order_factor_m = [1, None]
        z_order_factor_n = [1, None]

        print(f"z_order_factor_m={z_order_factor_m}, z_order_factor_n={z_order_factor_n}")

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                (z_order_factor_m[0] or z_order_factor_m[1]) * block_m,
                (z_order_factor_n[0] or z_order_factor_n[1]) * block_n,
                block_k,
            ],
        )

        # Step 3. Schedule matmul to use tensor core

        # inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, micro_size_m])
        j, j_inner = sch.split(j, factors=[None, micro_size_n])
        k, k_inner = sch.split(k, factors=[None, micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)

        block_inner = main_block
        block_outer = sch.blockize(i_inner)

        # split factors for i, j, and k
        in_wrap_block_cnt_m = block_m // thread_z // micro_size_m
        in_wrap_block_cnt_n = block_n // thread_y // micro_size_n
        in_wrap_block_cnt_k = block_k // micro_size_k

        i_factors = z_order_factor_m + [thread_z, in_wrap_block_cnt_m]
        j_factors = z_order_factor_n + [thread_y, in_wrap_block_cnt_n]
        k_factors = [None, in_wrap_block_cnt_k]

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, factors=k_factors)

        sch.reorder(i0, j0, i1, j1, k0, i2, j2, k1, i3, j3)
        block_axis = sch.fuse(batch, i0, j0, i1, j1)

        sch.bind(block_axis, "blockIdx.x")
        sch.bind(i2, "threadIdx.z")
        sch.bind(j2, "threadIdx.y")

        def fetch_input(block_outer, read_buffer_idx, read_loop_ndim, block_sizes, wmma_name):
            block_read = sch.cache_read(block_outer, read_buffer_idx, "shared.dyn")
            sch.compute_at(block_read, k0)
            fused = sch.fuse(*sch.get_loops(block_read)[-read_loop_ndim:])

            f0, f1, f2, f3, f4 = sch.split(
                fused, [None, thread_z, thread_y, warp_size, vector_size]
            )

            block_m, block_k, micro_size_m, micro_size_k = block_sizes

            sch.transform_layout(
                block_read,
                ("write", 0),
                lambda v0, v1, v2: (
                    v1 // block_m,
                    v2 // block_k,
                    v1 % block_m // micro_size_m,
                    v2 % block_k // micro_size_k,
                    v1 % micro_size_m,
                    v2 % micro_size_k,
                ),
            )

            sch.bind(f1, "threadIdx.z")
            sch.bind(f2, "threadIdx.y")
            sch.bind(f3, "threadIdx.x")
            sch.vectorize(f4)
            # sch.storage_align(block_read, 0, axis=-2, factor=16, offset=8)

            auto_inline_producers(sch, block_read)

            wmma_read = sch.cache_read(block_outer, read_buffer_idx, wmma_name)
            sch.compute_at(wmma_read, k1)
            return wmma_read

        wmma_read_a = fetch_input(
            block_outer, 0, 2, [block_m, block_k, micro_size_m, micro_size_k], "wmma.matrix_a"
        )
        wmma_read_b = fetch_input(
            block_outer, 1, 2, [block_n, block_k, micro_size_n, micro_size_k], "wmma.matrix_b"
        )

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        def store_output(block_outer, write_buffer_idx, write_loop_ndim, block_sizes, wmma_name):
            block_write = sch.cache_write(block_outer, write_buffer_idx, "shared.dyn")
            sch.reverse_compute_at(block_write, block_axis)

            fused = sch.fuse(*sch.get_loops(block_write)[-write_loop_ndim:])

            f0, f1, f2, f3, f4 = sch.split(
                fused, [None, thread_z, thread_y, warp_size, vector_size]
            )

            block_m, block_n, micro_size_m, micro_size_n = block_sizes
            sch.transform_layout(
                block_write,
                ("read", 0),
                lambda v0, v1, v2: (
                    v1 // block_m,
                    v2 // block_n,
                    v1 % block_m // micro_size_m,
                    v2 % block_n // micro_size_n,
                    v1 % micro_size_m,
                    v2 % micro_size_n,
                ),
            )

            sch.bind(f1, "threadIdx.z")
            sch.bind(f2, "threadIdx.y")
            sch.bind(f3, "threadIdx.x")
            sch.vectorize(f4)
            auto_inline_consumers(sch, block_write)

            store = sch.cache_write(block_outer, write_buffer_idx, wmma_name)
            v0, v1, v2, v3, v4, v5 = sch.get_loops(store)[-6:]
            v21, v22 = sch.split(v2, factors=[thread_z, None])
            v31, v32 = sch.split(v3, factors=[thread_z, None])
            sch.reorder(v21, v31, v22, v32)
            sch.bind(v21, "threadIdx.z")
            sch.bind(v31, "threadIdx.y")
            return store

        store = store_output(
            block_outer, 0, 2, [block_m, block_n, micro_size_m, micro_size_n], "wmma.accumulator"
        )

        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # unroll k
        # sch.annotate(k0, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        # sch.unroll(k0)
        # sch.annotate(k0, ann_key="pragma_unroll_explicit", ann_val=0)
        # k00, k01 = sch.split(k0, factors=[None, 8])
        # sch.annotate(k01, ann_key="pragma_unroll_explicit", ann_val=0)
        # sch.unroll(k01)

        # Tensorization by hardware intrinsics
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_wmma_intrin_group,
        )

        intrin_group = get_wmma_intrin_group(
            load_scope="shared.dyn",
            store_scope="shared.dyn",
            in_dtype=str(dtype_a),
            out_dtype=str(dtype_c),
            trans_b=True,
        )

        sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
        sch.tensorize(sch.get_loops(wmma_read_a)[-2], intrin_group["load_a"])
        sch.tensorize(sch.get_loops(wmma_read_b)[-2], intrin_group["load_b"])
        sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])
        sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])

        # async pipeline
        sch.annotate(k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 3])
        sch.annotate(k0, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
        sch.annotate(k0, ann_key="software_pipeline_async_stages", ann_val=[0])

        return sch


class Matmul(ScheduleRule):
    """The schedule rule for matmul-like computation"""

    @dataclass
    class Config:
        block_size_x: int = 8
        block_size_y: int = 8
        vthread_x: int = 1
        vthread_y: int = 1
        micro_size_x: int = 4
        micro_size_y: int = 4
        micro_size_k: int = 8
        vector_size: int = 1
        unroll: int = 256  # 0 means no unroll
        use_shared: bool = True
        storage_align: bool = False
        inner_x: bool = False

    def get_configs(self, target: Target) -> Config:
        """Get the schedule config for the target"""
        if target.kind.name == "cuda" or target.kind.name == "rocm":
            return Matmul.Config(
                block_size_x=8,
                block_size_y=16,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=4,
                micro_size_y=4,
                micro_size_k=16,
                vector_size=2,
                unroll=256,
                use_shared=True,
                storage_align=True,
                inner_x=False,
            )
        elif target.kind.name == "opencl" and "android" in str(target.host):
            return Matmul.Config(
                block_size_x=8,
                block_size_y=8,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=8,
                micro_size_y=2,
                micro_size_k=16,
                vector_size=8,
                unroll=64,
                use_shared=False,
                storage_align=False,
                inner_x=True,
            )
        else:
            return Matmul.Config()

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        index_maps = get_index_map(block_stmt)
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Step 0. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 1. Check Tensor Core support
        # Tensorization config:
        # If any value of I, J, K is fixed and less than this threshold,
        # tensorization rule will not be applied.
        minimal_tensorize_threshold = 64
        block_stmt = sch.get(main_block)
        if target.kind.name == "cuda" and check_sm_version(target.arch) >= 70:
            apply_tensorization: bool = True
            # the batch dimension is not taken into consideration.
            for item_var in block_stmt.iter_vars[1:]:
                extent = item_var.dom.extent
                if isinstance(extent, tir.expr.IntImm):
                    if extent.value <= minimal_tensorize_threshold:
                        apply_tensorization = False
            if apply_tensorization:
                tensorize_sch = MatmulTensorization().apply(func, target, _)
                if tensorize_sch is not None:
                    return tensorize_sch

        # Step 2. Get schedule config.
        config = self.get_configs(target)

        # Step 3. Schedule matmul
        y_kernel_size = config.vthread_y * config.block_size_y * config.micro_size_y
        x_kernel_size = config.vthread_x * config.block_size_x * config.micro_size_x
        if config.inner_x:
            sch.pad_einsum(
                main_block,
                [1, y_kernel_size, x_kernel_size, config.micro_size_k],
            )
            batch, y, x, k = sch.get_loops(main_block)
        else:
            sch.pad_einsum(
                main_block,
                [1, x_kernel_size, y_kernel_size, config.micro_size_k],
            )
            batch, x, y, k = sch.get_loops(main_block)
        by, vy, ty, yi = sch.split(
            y, [None, config.vthread_y, config.block_size_y, config.micro_size_y]
        )
        bx, vx, tx, xi = sch.split(
            x, [None, config.vthread_x, config.block_size_x, config.micro_size_x]
        )
        ko, ki = sch.split(k, factors=[None, config.micro_size_k])
        sch.reorder(by, bx, vy, vx, ty, tx, ko, ki, yi, xi)
        by = sch.fuse(batch, by)
        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        inner_loop = config.micro_size_x if config.inner_x else config.micro_size_y
        if inner_loop % config.vector_size == 0:
            _, v = sch.split(xi, [None, config.vector_size])
            sch.vectorize(v)

        if config.unroll > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=config.unroll)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)

        l2g = sch.cache_write(main_block, 0, "local")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        if config.micro_size_x % config.vector_size == 0:
            _, v = sch.split(sch.get_loops(l2g)[-1], [None, config.vector_size])
            sch.vectorize(v)

        if config.use_shared:

            def _cooperative_fetch(index, vec_len):
                block = sch.cache_read(main_block, index, "shared")
                num_loops = len(sch.get_loops(block))
                sch.compute_at(block, ko, preserve_unit_loops=True)
                loops = sch.get_loops(block)[-num_loops:]
                ty, tx, _, vec = sch.split(
                    sch.fuse(*loops),
                    factors=[config.block_size_y, config.block_size_x, None, vec_len],
                )
                sch.vectorize(vec)
                sch.bind(ty, "threadIdx.y")
                sch.bind(tx, "threadIdx.x")
                if config.storage_align:
                    sch.storage_align(block, 0, axis=1, factor=8, offset=vec_len)
                return block

            a_g2s = _cooperative_fetch(0, vec_len=config.vector_size)
            b_g2s = _cooperative_fetch(1, vec_len=config.vector_size)

            auto_inline_producers(sch, a_g2s)
            auto_inline_producers(sch, b_g2s)
        else:
            auto_inline_producers(sch, main_block)

        auto_inline_consumers(sch, l2g)

        remaining_consumers = sch.get_consumers(l2g)

        if len(remaining_consumers) != 0:
            # Some blocks have failed to be inlined to the producer cache-write stage.
            # This could be due to another producer block that has not been scheduled.
            for c in remaining_consumers:
                for p in sch.get_producers(c):
                    if sch.get(p) != sch.get(l2g):
                        sch.compute_inline(p)

            # Try inlining into the cache-write stage again, this time it should succeed.
            auto_inline_consumers(sch, l2g)

        # msg = "There are some consumers of the cache-write stage that are not properly inlined."
        # assert len(sch.get_consumers(l2g)) == 0, msg
        # Step4. Check if there are unbound blocks. Execute fallback scheduling to them.
        def is_scheduled(block: tir.schedule.BlockRV) -> bool:
            loops = sch.get_loops(block)
            loop_kinds = {sch.get(loop).kind for loop in loops}
            return loop_kinds != {ForKind.SERIAL}

        blocks = sch.get_child_blocks(root_block)
        max_threads_per_block = utils.max_threads_per_block(target)
        for block in blocks:
            if is_scheduled(block):
                continue
            # no axis of the block is bound to thread or block
            s_loops = sch.get_loops(block)
            bx, tx = sch.split(  # pylint: disable=invalid-name
                sch.fuse(*s_loops),
                factors=[None, max_threads_per_block],
            )
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")

        sch.decompose_reduction(main_block, ko)
        return sch
