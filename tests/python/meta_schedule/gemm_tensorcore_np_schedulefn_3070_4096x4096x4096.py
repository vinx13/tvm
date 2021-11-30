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
"""Integration test for CUDA with Tensor Core"""
# pylint: disable=missing-function-docstring
import pytest
import te_workload
import tvm
import tir_tensor_intrin  # pylint: disable=unused-import
from tvm import te, tir
from tvm import meta_schedule as ms
import tvm.testing
import numpy as np
import os
from tvm.contrib import nvcc
import sys

TARGET = tvm.target.Target("nvidia/geforce-rtx-3070")

TASK = "gemm"
USE_MANUAL_CODE = False


@tvm.register_func('tvm_callback_cuda_compile', override=True)
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx")
    return ptx


def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code


def test_integration_matmul():
    N = 4096
    M = 4096
    K = 4096
    workload = te_workload.matmul_fp16(n=N, m=M, k=K)
    workload = te.create_prim_func(workload)

    def schedule(sch: tir.Schedule):
        block = sch.get_block("C")
        i, j, k = sch.get_loops(block)
        # Step 1. Rule-Auto-Tensorize
        # pylint: disable=invalid-name
        i, i_tc = sch.split(i, factors=[None, 16])
        j, j_tc = sch.split(j, factors=[None, 16])
        k, k_tc = sch.split(k, factors=[None, 16])
        sch.reorder(
            # fmt: off
            i, j, k,
            # tensor core
            i_tc, j_tc, k_tc,
            # fmt: on
        )
        block_inner = sch.blockize(i_tc)
        block_outer, block_inner = block_inner, block
        del block
        # Step 2. Rule-Multi-Level-Tiling
        # i_factors = sch.sample_perfect_tile(i, n=5, decision=[8, 1, 2, 1, 4])
        # j_factors = sch.sample_perfect_tile(j, n=5, decision=[1, 8, 4, 1, 2])
        i_factors = [64, 1, 2, 1, 2]
        j_factors = [8, 4, 2, 1, 4]  # swizzle: identity8
        j_factors = [1, 32, 2, 1, 4] 
        k_factors = [128, 2, 1]
        i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
        k0, k1, k2 = sch.split(k, k_factors)
        # pylint: enable=invalid-name
        sch.reorder(
            # fmt: off
            i0, j0,   # S => blockIdx.x
            i1, j1,   # S => blockIdx.y
            j2, i2,   # S => threadIdx.y
            # cache_write here
            k0,       # R
            # vectorized cooperative fetching here
            k1,       # R
            i3, j3,   # S
            k2,       # R
            i4, j4,
            # S
            # fmt: on
        )
        block_idx = sch.fuse(i0, j0)
        block_idy = sch.fuse(i1, j1)
        thread_idy = sch.fuse(j2, i2)
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")

        # num_ty = sch.get(i_factors[2]) * sch.get(j_factors[2])
        num_ty = i_factors[2] * j_factors[2]
        def fetch_to_shared(block, idx, ndim):
            # shared [128 x 32]
            block_read = sch.cache_read(block, idx, "shared")
            # block_read_local = sch.cache_read(block_read, 0, "local")
            sch.compute_at(block_read, k0)
            vector_size = 8
            warp_size = 32
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
            f_0, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])
            # sch.reorder(f_1, f_2, f_0, f_3)
            sch.bind(f_2, 'threadIdx.x')
            sch.bind(f_1, 'threadIdx.y')
            # sch.bind(f_0, 'vthread.z')
            # sch.compute_at(block_read_local, f_2)
            sch.vectorize(f_3)

            sch.storage_align(block_read, 0, axis=-2, factor=32, offset=8)


        fetch_to_shared(block_outer, 1, 2)

        fetch_to_shared(block_outer, 2, 2)

        # Step 3. Postproc-Rewrite-Tensorize
        # Step 3.1. Cache read
        loop = sch.get_loops(block_outer)[-1]
        block_read_a = sch.cache_read(block_outer, 1, "wmma.matrix_a")
        block_read_b = sch.cache_read(block_outer, 2, "wmma.matrix_b")
        sch.compute_at(block_read_a, k1)
        sch.compute_at(block_read_b, k1)
        # Step 3.2. Cache write
        block_write_c = sch.cache_write(block_outer, 0, "wmma.accumulator")
        # block_outer, block_write_c = block_write_c, block_outer
        sch.reverse_compute_at(block_write_c, thread_idy)
        # Wuwei: we also need spliting the write back stage.
        ii, jj = sch.get_loops(block_write_c)[-2:]
        io, ii = sch.split(ii, factors=[None, 16])
        jo, ji = sch.split(jj, factors=[None, 16])
        sch.reorder(io, jo, ii, ji)
        # Step 3.3. Decompose
        loop = sch.get_loops(block_outer)[3]
        block_init_c = sch.decompose_reduction(block_outer, loop)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]
        # Step 3.4. Tensorize
        loop = sch.get_loops(block_inner)[-3]
        # print(tvm.script.asscript(sch.mod['main']))
        def tile_wmma_fragment(block_read):
            i, j = sch.get_loops(block_read)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            return i1

        sch.tensorize(loop, "wmma_sync")
        loop = tile_wmma_fragment(block_read_a)
        sch.tensorize(loop, "wmma_load_a")
        loop = tile_wmma_fragment(block_read_b)
        sch.tensorize(loop, "wmma_load_b")
        loop = sch.get_loops(block_init_c_inner)[-2]
        sch.tensorize(loop, "wmma_fill")
        loop = sch.get_loops(block_write_c)[-2]
        sch.tensorize(loop, "wmma_store")

    # task = ms.SearchTask(
    #         workload=workload,
    #         target=TARGET,
    #         target_host='llvm',
    #         task_name="cuda_matmul",
    #         log_file="./cuda_matmul.json",
    #     )
    # space = ms.space.ScheduleFn(
    #     schedule,
    #     postprocs=[
    #         ms.postproc.verify_gpu_code(),
    #     ],
    # )
    # Evolutionary search doesn't support using result of sch.get() as the split factor.
    # Enable this when we have postprocessors for auto tensorization.
    # evolutionary = ms.strategy.Evolutionary(
    #         total_measures=256,
    #         num_measures_per_iter=16,
    #         population=128,
    #         init_measured_ratio=0.2,
    #         genetic_algo_iters=10,
    #         p_mutate=0.85,
    #         mutator_probs={
    #             ms.mutator.mutate_tile_size(): 1.0,
    #         },
    #         cost_model=ms.XGBModel(
    #             num_warmup_samples=0,
    #         ),
    #         eps_greedy=0.05,
    #     )
    sch = tir.Schedule(workload)
    schedule(sch)

    # replay = ms.strategy.Replay(256)
    # sch = ms.autotune(
    #     task=task,
    #     space=space,
    #     strategy=replay,
    #     measurer=ms.ProgramMeasurer(
    #         measure_callbacks=[
    #             ms.RecordToFile(),
    #         ]
    #     ),
    # )
    # assert space.postprocess(task, sch)
    if sch is None:
        print("No valid schedule found")
    else:
        print(sch.mod['main'].script())
        print(tvm.lower(sch.mod['main'], None, simple_mode=True))

    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(N, K)).astype("float16")
    b_np = np.random.uniform(size=(K, M)).astype("float16")
    c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((N, M), dtype="float32"), dev)
    # sys.exit(0)
    f = tvm.build(sch.mod['main'], target="cuda", name="dense")
    print(f.imported_modules[0].get_source())
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
    gflops = (N*M*K) * 2 / 1e9
    time_ms = evaluator(a, b, c).mean * 1e3
    print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))


if __name__ == "__main__":
    test_integration_matmul()
