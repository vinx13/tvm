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
# pylint: disable=missing-function-docstring,missing-module-docstring
import sys

import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


@T.prim_func
def transformed_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with T.block("update"):
            vi, vj = T.axis.remap("SS", [i0, i1])
            vk = T.axis.R(128, i2_outer * 32 + i2_inner_outer * 4 + i2_inner_inner)
            T.reads([A[vi, vk], B[vj, vk]])
            T.writes([C[vi, vj]])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@T.prim_func
def matmul_rfactor(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(128, 128), "float32"],
    C: T.Buffer[(128, 128), "float32"],
) -> None:
    C_rf = T.alloc_buffer([4, 128, 128], dtype="float32")
    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with T.block("update_rf"):
            vi, vj, vi2_outer, vi2_inner_outer, vi2_inner_inner = T.axis.remap(
                "SSRRS", [i0, i1, i2_outer, i2_inner_outer, i2_inner_inner]
            )
            with T.init():
                C_rf[vi2_inner_inner, vi, vj] = T.float32(0)
            C_rf[vi2_inner_inner, vi, vj] = (
                C_rf[vi2_inner_inner, vi, vj]
                + A[vi, vi2_outer * 32 + vi2_inner_outer * 4 + vi2_inner_inner]
                * B[vj, vi2_outer * 32 + vi2_inner_outer * 4 + vi2_inner_inner]
            )
    for i0, i1, i2_inner_inner in T.grid(128, 128, 4):
        with T.block("update"):
            vi, vj, vi2_inner_inner = T.axis.remap("SSR", [i0, i1, i2_inner_inner])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + C_rf[vi2_inner_inner, vi, vj]


@T.prim_func
def matmul_not_stage_pipeline(a: T.handle, b: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [256, 256])
    B = T.match_buffer(b, [256, 256])
    D = T.match_buffer(d, [256, 256])
    C = T.alloc_buffer([256, 256])

    for i, j, k in T.grid(128, 128, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    for i, j in T.grid(256, 256):
        with T.block("D"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = C[vi, vj]


@T.prim_func
def matmul_not_same_buffer_access(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))

    for i, j, k in T.grid(128, 128, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vj, vi] = C[vj, vi] + A[vi, vk] * B[vk, vj]


@T.prim_func
def matmul_loop_multiple_children(a: T.handle, b: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    D = T.match_buffer(d, [128, 128])

    for k, i, j in T.grid(128, 128, 128):
        with T.block("C"):
            ck, ci, cj = T.axis.remap("RSS", [k, i, j])
            with T.init():
                C[ci, cj] = 0.0
            C[ci, cj] = C[ci, cj] + A[ci, ck] * B[ck, cj]
        with T.block("D"):
            dk, di, dj = T.axis.remap("RSS", [k, i, j])
            with T.init():
                D[di, dj] = 0.0
            D[di, dj] = D[di, dj] + B[di, dk] * A[dk, dj]


@T.prim_func
def square_sum(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    C = T.match_buffer(c, [16])

    for b0, i0, j0 in T.grid(16, 256, 256):
        with T.block("C"):
            b, i, j = T.axis.remap("SRR", [b0, i0, j0])
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + A[b, i, j] * A[b, i, j]


@T.prim_func
def square_sum_rfactor(
    A: T.Buffer[(16, 256, 256), "float32"], C: T.Buffer[(16,), "float32"]
) -> None:
    C_rf = T.alloc_buffer([16, 256], dtype="float32")
    for b0, i0, j0 in T.grid(16, 256, 256):
        with T.block("C_rf"):
            b, i, vj0 = T.axis.remap("SRS", [b0, i0, j0])
            with T.init():
                C_rf[b, vj0] = T.float32(0)
            C_rf[b, vj0] = C_rf[b, vj0] + A[b, i, vj0] * A[b, i, vj0]
    for b0, j0 in T.grid(16, 256):
        with T.block("C"):
            b, vj0 = T.axis.remap("SR", [b0, j0])
            with T.init():
                C[b] = T.float32(0)
            C[b] = C[b] + C_rf[b, vj0]


@T.prim_func
def transformed_square_sum_square_root(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    D = T.match_buffer(d, [16])
    C = T.alloc_buffer([16])

    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 32768, 2):
        with T.block("C"):
            b = T.axis.S(16, i0)
            i = T.axis.R(256, T.floordiv(i1_i2_fused_outer, 256))
            j = T.axis.R(256, T.floormod(i1_i2_fused_outer, 256))
            T.reads([A[b, i, j]])
            T.writes([C[b]])
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + (A[b, i, j] * A[b, i, j])
    for i0_1 in T.serial(0, 16):
        with T.block("D"):
            b_1 = T.axis.S(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def square_sum_square_root_rfactor(
    A: T.Buffer[(16, 256, 256), "float32"], D: T.Buffer[(16,), "float32"]
) -> None:
    C = T.alloc_buffer([16], dtype="float32")
    C_rf = T.alloc_buffer([2, 16], dtype="float32")
    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 32768, 2):
        with T.block("C_rf"):
            b, vi1_i2_fused_outer, vi1_i2_fused_inner = T.axis.remap(
                "SRS", [i0, i1_i2_fused_outer, i1_i2_fused_inner]
            )
            with T.init():
                C_rf[vi1_i2_fused_inner, b] = T.float32(0)
            C_rf[vi1_i2_fused_inner, b] = (
                C_rf[vi1_i2_fused_inner, b]
                + A[
                    b,
                    (vi1_i2_fused_outer * 2 + vi1_i2_fused_inner) // 256,
                    (vi1_i2_fused_outer * 2 + vi1_i2_fused_inner) % 256,
                ]
                * A[
                    b,
                    (vi1_i2_fused_outer * 2 + vi1_i2_fused_inner) // 256,
                    (vi1_i2_fused_outer * 2 + vi1_i2_fused_inner) % 256,
                ]
            )
    for i0, i1_i2_fused_inner in T.grid(16, 2):
        with T.block("C"):
            b, vi1_i2_fused_inner = T.axis.remap("SR", [i0, i1_i2_fused_inner])
            with T.init():
                C[b] = T.float32(0)
            C[b] = C[b] + C_rf[vi1_i2_fused_inner, b]
    for i0_1 in T.serial(16):
        with T.block("D"):
            b_1 = T.axis.spatial(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def transformed_square_sum_square_root_factor_one_1(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    D = T.match_buffer(d, [16])
    C = T.alloc_buffer([16])

    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 65536, 1):
        with T.block("C"):
            b = T.axis.S(16, i0)
            i = T.axis.R(256, T.floordiv(i1_i2_fused_outer, 256))
            j = T.axis.R(256, T.floormod(i1_i2_fused_outer, 256))
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + (A[b, i, j] * A[b, i, j])
    for i0_1 in T.serial(0, 16):
        with T.block("D"):
            b_1 = T.axis.S(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def square_sum_square_root_factor_one_1_rfactor(
    A: T.Buffer[(16, 256, 256), "float32"], D: T.Buffer[(16,), "float32"]
) -> None:
    C = T.alloc_buffer([16], dtype="float32")
    C_rf = T.alloc_buffer([1, 16], dtype="float32")
    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 65536, 1):
        with T.block("C_rf"):
            b = T.axis.spatial(16, i0)
            i = T.axis.reduce(256, i1_i2_fused_outer // 256)
            j = T.axis.reduce(256, i1_i2_fused_outer % 256)
            vi1_i2_fused_inner = T.axis.spatial(1, i1_i2_fused_inner)
            with T.init():
                C_rf[vi1_i2_fused_inner, b] = T.float32(0)
            C_rf[vi1_i2_fused_inner, b] = C_rf[vi1_i2_fused_inner, b] + A[b, i, j] * A[b, i, j]
    for i0, i1_i2_fused_inner in T.grid(16, 1):
        with T.block("C"):
            b, vi1_i2_fused_inner = T.axis.remap("SR", [i0, i1_i2_fused_inner])
            with T.init():
                C[b] = T.float32(0)
            C[b] = C[b] + C_rf[vi1_i2_fused_inner, b]
    for i0_1 in T.serial(16):
        with T.block("D"):
            b_1 = T.axis.spatial(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def transformed_square_sum_square_root_factor_one_2(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    D = T.match_buffer(d, [16])
    C = T.alloc_buffer([16])

    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 1, 65536):
        with T.block("C"):
            b = T.axis.S(16, i0)
            i = T.axis.R(256, T.floordiv(i1_i2_fused_inner, 256))
            j = T.axis.R(256, T.floormod(i1_i2_fused_inner, 256))
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + (A[b, i, j] * A[b, i, j])
    for i0_1 in T.serial(0, 16):
        with T.block("D"):
            b_1 = T.axis.S(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def square_sum_square_root_factor_one_2_rfactor(
    A: T.Buffer[(16, 256, 256), "float32"], D: T.Buffer[(16,), "float32"]
) -> None:
    C = T.alloc_buffer([16], dtype="float32")
    C_rf = T.alloc_buffer([16, 1], dtype="float32")
    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 1, 65536):
        with T.block("C_rf"):
            b = T.axis.spatial(16, i0)
            i = T.axis.reduce(256, i1_i2_fused_inner // 256)
            j = T.axis.reduce(256, i1_i2_fused_inner % 256)
            vi1_i2_fused_outer = T.axis.spatial(1, i1_i2_fused_outer)
            with T.init():
                C_rf[b, vi1_i2_fused_outer] = T.float32(0)
            C_rf[b, vi1_i2_fused_outer] = C_rf[b, vi1_i2_fused_outer] + A[b, i, j] * A[b, i, j]
    for i0, i1_i2_fused_outer in T.grid(16, 1):
        with T.block("C"):
            b, vi1_i2_fused_outer = T.axis.remap("SR", [i0, i1_i2_fused_outer])
            with T.init():
                C[b] = T.float32(0)
            C[b] = C[b] + C_rf[b, vi1_i2_fused_outer]
    for i0_1 in T.serial(16):
        with T.block("D"):
            b_1 = T.axis.spatial(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def transformed_square_sum_square_root_factor_one_1(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    D = T.match_buffer(d, [16])
    C = T.alloc_buffer([16])

    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 65536, 1):
        with T.block("C"):
            b = T.axis.S(16, i0)
            i = T.axis.R(256, T.floordiv(i1_i2_fused_outer, 256))
            j = T.axis.R(256, T.floormod(i1_i2_fused_outer, 256))
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + (A[b, i, j] * A[b, i, j])
    for i0_1 in T.serial(0, 16):
        with T.block("D"):
            b_1 = T.axis.S(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def square_sum_square_root_factor_one_1_rfactor(
    A: T.Buffer[(16, 256, 256), "float32"], D: T.Buffer[(16,), "float32"]
) -> None:
    C = T.alloc_buffer([16], dtype="float32")
    C_rf = T.alloc_buffer([1, 16], dtype="float32")
    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 65536, 1):
        with T.block("C_rf"):
            b = T.axis.spatial(16, i0)
            i = T.axis.reduce(256, i1_i2_fused_outer // 256)
            j = T.axis.reduce(256, i1_i2_fused_outer % 256)
            vi1_i2_fused_inner = T.axis.spatial(1, i1_i2_fused_inner)
            with T.init():
                C_rf[vi1_i2_fused_inner, b] = T.float32(0)
            C_rf[vi1_i2_fused_inner, b] = C_rf[vi1_i2_fused_inner, b] + A[b, i, j] * A[b, i, j]
    for i0, i1_i2_fused_inner in T.grid(16, 1):
        with T.block("C"):
            b, vi1_i2_fused_inner = T.axis.remap("SR", [i0, i1_i2_fused_inner])
            with T.init():
                C[b] = T.float32(0)
            C[b] = C[b] + C_rf[vi1_i2_fused_inner, b]
    for i0_1 in T.serial(16):
        with T.block("D"):
            b_1 = T.axis.spatial(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def transformed_square_sum_square_root_factor_one_2(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    D = T.match_buffer(d, [16])
    C = T.alloc_buffer([16])

    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 1, 65536):
        with T.block("C"):
            b = T.axis.S(16, i0)
            i = T.axis.R(256, T.floordiv(i1_i2_fused_inner, 256))
            j = T.axis.R(256, T.floormod(i1_i2_fused_inner, 256))
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + (A[b, i, j] * A[b, i, j])
    for i0_1 in T.serial(0, 16):
        with T.block("D"):
            b_1 = T.axis.S(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def square_sum_square_root_factor_one_2_rfactor(
    A: T.Buffer[(16, 256, 256), "float32"], D: T.Buffer[(16,), "float32"]
) -> None:
    C = T.alloc_buffer([16], dtype="float32")
    C_rf = T.alloc_buffer([16, 1], dtype="float32")
    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 1, 65536):
        with T.block("C_rf"):
            b = T.axis.spatial(16, i0)
            i = T.axis.reduce(256, i1_i2_fused_inner // 256)
            j = T.axis.reduce(256, i1_i2_fused_inner % 256)
            vi1_i2_fused_outer = T.axis.spatial(1, i1_i2_fused_outer)
            with T.init():
                C_rf[b, vi1_i2_fused_outer] = T.float32(0)
            C_rf[b, vi1_i2_fused_outer] = C_rf[b, vi1_i2_fused_outer] + A[b, i, j] * A[b, i, j]
    for i0, i1_i2_fused_outer in T.grid(16, 1):
        with T.block("C"):
            b, vi1_i2_fused_outer = T.axis.remap("SR", [i0, i1_i2_fused_outer])
            with T.init():
                C[b] = T.float32(0)
            C[b] = C[b] + C_rf[b, vi1_i2_fused_outer]
    for i0_1 in T.serial(16):
        with T.block("D"):
            b_1 = T.axis.spatial(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def square_sum_with_annotation(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    C = T.match_buffer(c, [16])

    for b0, i0, j0 in T.grid(16, 256, 256):
        with T.block("C"):
            T.block_attr({"test_annotation": 1})
            b, i, j = T.axis.remap("SRR", [b0, i0, j0])
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + A[b, i, j] * A[b, i, j]


@T.prim_func
def square_sum_with_annotation_rfactor(
    A: T.Buffer[(16, 256, 256), "float32"], C: T.Buffer[(16,), "float32"]
) -> None:
    C_rf = T.alloc_buffer([16, 256], dtype="float32")
    for b0, i0, j0 in T.grid(16, 256, 256):
        with T.block("C_rf"):
            b, i, vj0 = T.axis.remap("SRS", [b0, i0, j0])
            T.block_attr({"test_annotation": 1})
            with T.init():
                C_rf[b, vj0] = T.float32(0)
            C_rf[b, vj0] = C_rf[b, vj0] + A[b, i, vj0] * A[b, i, vj0]
    for b0, j0 in T.grid(16, 256):
        with T.block("C"):
            b, vj0 = T.axis.remap("SR", [b0, j0])
            T.block_attr({"test_annotation": 1})
            with T.init():
                C[b] = T.float32(0)
            C[b] = C[b] + C_rf[b, vj0]


@T.prim_func
def element_wise(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def rowsum(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i, k in T.grid(128, 128):
        with T.block("B"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_not_quasi_affine(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i, k in T.grid(128, 16):
        with T.block("B"):
            vi = T.axis.S(128, i)
            vk = T.axis.R(128, T.floordiv(k * k, 2))
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_not_dominant(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))

    for i, k in T.grid(128, 128):
        with T.block("B"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                B[vi, vk] = 0.0
            B[vi, vk] = B[vi, vk] + A[vi, vk]


@T.prim_func
def rowsum_not_serial(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i in T.serial(0, 128):
        for k in T.parallel(0, 128):
            with T.block("B"):
                vi, vk = T.axis.remap("SR", [i, k])
                with T.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_wrong_reduce_pattern1(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i, k in T.grid(128, 128):
        with T.block("B"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                B[vi] = 1.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_wrong_reduce_pattern2(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i, k in T.grid(128, 128):
        with T.block("B"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] - A[vi, vk]


@T.prim_func
def rowsum_transformed(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for io, ii_ko_fused, ki in T.grid(32, 128, 4):
        with T.block("B"):
            vi = T.axis.S(128, io * 4 + T.floordiv(ii_ko_fused, 32))
            vk = T.axis.R(128, T.floormod(ii_ko_fused, 32) * 4 + ki)
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_zero_dim(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128])
    B = T.match_buffer(b, [])

    for k0 in range(128):
        with T.block("B"):
            k = T.axis.R(128, k0)
            with T.init():
                B[()] = 0.0
            B[()] = B[()] + A[k]


@T.prim_func
def rowsum_zero_dim_rfactor(A: T.Buffer[(128,), "float32"], B: T.Buffer[(), "float32"]) -> None:
    B_rf = T.alloc_buffer([128], dtype="float32")
    for k0 in T.serial(128):
        with T.block("B_rf"):
            vi0 = T.axis.S(128, i)
            B_rf[vi0] = A[vi0]

    for i in range(128):
        with T.block("B"):
            vk0 = T.axis.reduce(128, k0)
            with T.init():
                B[()] = T.float32(0)
            B[()] = B[()] + B_rf[vk0]


@T.prim_func
def rowsum_predicate(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    for i, k_0, k_1 in T.grid(128, 13, 10):
        with T.block("B"):
            T.where(k_0 * 10 + k_1 < 128)
            vi = T.axis.S(128, i)
            vk = T.axis.R(128, k_0 * 10 + k_1)
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_predicate_rfactor(
    A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128,), "float32"]
) -> None:
    B_rf = T.alloc_buffer([128, 13], dtype="float32")
    for i, k_0, k_1 in T.grid(128, 13, 10):
        with T.block("B_rf"):
            vi, vk_0, vk_1 = T.axis.remap("SSR", [i, k_0, k_1])
            T.where(k_0 * 10 + k_1 < 128)
            with T.init():
                B_rf[vi, vk_0] = T.float32(0)
            B_rf[vi, vk_0] = B_rf[vi, vk_0] + A[vi, vk_0 * 10 + vk_1]
    for i, k_0 in T.grid(128, 13):
        with T.block("B"):
            vi, vk_0 = T.axis.remap("SR", [i, k_0])
            with T.init():
                B[vi] = T.float32(0)
            B[vi] = B[vi] + B_rf[vi, vk_0]


@T.prim_func
def multiple_reduction_blocks(a: T.handle, f: T.handle) -> None:
    A = T.match_buffer(a, (16, 16, 16))
    C = T.alloc_buffer((16, 16))
    D = T.alloc_buffer((16, 16))
    E = T.alloc_buffer((16, 16))
    F = T.match_buffer(f, (16, 16))

    for i in T.serial(0, 16):
        for j1 in T.serial(0, 16):
            for k1o, k1i in T.grid(4, 4):
                with T.block("C"):
                    ci, cj = T.axis.remap("SS", [i, j1])
                    ck = T.axis.R(16, k1o * 4 + k1i)
                    with T.init():
                        C[ci, cj] = 0.0
                    C[ci, cj] = C[ci, cj] + A[ci, cj, ck]
            for k2o, k2i in T.grid(4, 4):
                with T.block("D"):
                    di, dj = T.axis.remap("SS", [i, j1])
                    dk = T.axis.R(16, k2o * 4 + k2i)
                    with T.init():
                        D[di, dj] = 0.0
                    D[di, dj] = D[di, dj] + A[di, dj, dk] + C[di, dj]
        for j2 in T.serial(0, 16):
            for k3o, k3i in T.grid(4, 4):
                with T.block("E"):
                    ei, ej = T.axis.remap("SS", [i, j2])
                    ek = T.axis.R(16, k3o * 4 + k3i)
                    with T.init():
                        E[ei, ej] = 0.0
                    E[ei, ej] = E[ei, ej] + A[ei, ej, ek] + D[ei, ej]
            for k4o, k4i in T.grid(4, 4):
                with T.block("F"):
                    fi, fj = T.axis.remap("SS", [i, j2])
                    fk = T.axis.R(16, k4o * 4 + k4i)
                    with T.init():
                        F[fi, fj] = 0.0
                    F[fi, fj] = F[fi, fj] + A[fi, fj, fk] + E[fi, fj]


@T.prim_func
def multiple_reduction_blocks_rfactor(
    A: T.Buffer[(16, 16, 16), "float32"], F: T.Buffer[(16, 16), "float32"]
) -> None:
    C = T.alloc_buffer([16, 16], dtype="float32")
    D = T.alloc_buffer([16, 16], dtype="float32")
    E = T.alloc_buffer([16, 16], dtype="float32")
    C_rf = T.alloc_buffer([16, 16, 4], dtype="float32")
    for i, j1, k1o, k1i in T.grid(16, 16, 4, 4):
        with T.block("C_rf"):
            ci, cj, vk1o, vk1i = T.axis.remap("SSSR", [i, j1, k1o, k1i])
            with T.init():
                C_rf[ci, cj, vk1o] = T.float32(0)
            C_rf[ci, cj, vk1o] = C_rf[ci, cj, vk1o] + A[ci, cj, vk1o * 4 + vk1i]
    for i in T.serial(16):
        for j1 in T.serial(16):
            for k1o in T.serial(4):
                with T.block("C"):
                    ci, cj, vk1o = T.axis.remap("SSR", [i, j1, k1o])
                    with T.init():
                        C[ci, cj] = T.float32(0)
                    C[ci, cj] = C[ci, cj] + C_rf[ci, cj, vk1o]
            for k2o, k2i in T.grid(4, 4):
                with T.block("D"):
                    di, dj = T.axis.remap("SS", [i, j1])
                    dk = T.axis.reduce(16, k2o * 4 + k2i)
                    with T.init():
                        D[di, dj] = T.float32(0)
                    D[di, dj] = D[di, dj] + A[di, dj, dk] + C[di, dj]
        for j2 in T.serial(16):
            for k3o, k3i in T.grid(4, 4):
                with T.block("E"):
                    ei, ej = T.axis.remap("SS", [i, j2])
                    ek = T.axis.reduce(16, k3o * 4 + k3i)
                    with T.init():
                        E[ei, ej] = T.float32(0)
                    E[ei, ej] = E[ei, ej] + A[ei, ej, ek] + D[ei, ej]
            for k4o, k4i in T.grid(4, 4):
                with T.block("F"):
                    fi, fj = T.axis.remap("SS", [i, j2])
                    fk = T.axis.reduce(16, k4o * 4 + k4i)
                    with T.init():
                        F[fi, fj] = T.float32(0)
                    F[fi, fj] = F[fi, fj] + A[fi, fj, fk] + E[fi, fj]


@T.prim_func
def rfactor_spatial_only(
    A: T.Buffer[(1, 512, 7, 7), "float32"],
    B: T.Buffer[(1, 512, 1, 1), "float32"],
) -> None:
    for _i0, i1, _i2, _i3, i4, _i5 in T.grid(1, 512, 1, 1, 49, 1):
        with T.block("acc"):
            ax0 = T.axis.spatial(1, 0)
            ax1 = T.axis.spatial(512, i1)
            ax2 = T.axis.spatial(1, 0)
            ax3 = T.axis.spatial(1, 0)
            rv0 = T.axis.reduce(7, i4 // 7)
            rv1 = T.axis.reduce(7, i4 % 7)
            T.reads(A[ax0, ax1, ax2 * 7 + rv0, ax3 * 7 + rv1])
            T.writes(B[ax0, ax1, ax2, ax3])
            with T.init():
                B[ax0, ax1, ax2, ax3] = T.float32(0)
            B[ax0, ax1, ax2, ax3] = (
                B[ax0, ax1, ax2, ax3] + A[ax0, ax1, ax2 * 7 + rv0, ax3 * 7 + rv1]
            )


@T.prim_func
def rfactor_spatial_only_after(
    A: T.Buffer[(1, 512, 7, 7), "float32"],
    B: T.Buffer[(1, 512, 1, 1), "float32"],
) -> None:
    # body
    # with T.block("root")
    B_rf = T.alloc_buffer([1, 512, 1, 1, 49], dtype="float32")
    for _i0, i1, _i2, _i3, i4, _i5 in T.grid(1, 512, 1, 1, 49, 1):
        with T.block("acc_rf"):
            vi4 = T.axis.spatial(49, i4)
            ax0 = T.axis.spatial(1, 0)
            ax1 = T.axis.spatial(512, i1)
            ax2 = T.axis.spatial(1, 0)
            ax3 = T.axis.spatial(1, 0)
            B_rf[ax0, ax1, ax2, ax3, vi4] = A[ax0, ax1, ax2 * 7 + vi4 // 7, ax3 * 7 + vi4 % 7]
    for _i0, i1, _i2, _i3, i4, _i5 in T.grid(1, 512, 1, 1, 49, 1):
        with T.block("acc"):
            vi4 = T.axis.reduce(49, i4)
            ax0 = T.axis.spatial(1, 0)
            ax1 = T.axis.spatial(512, i1)
            ax2 = T.axis.spatial(1, 0)
            ax3 = T.axis.spatial(1, 0)
            with T.init():
                B[ax0, ax1, ax2, ax3] = T.float32(0)
            B[ax0, ax1, ax2, ax3] = B[ax0, ax1, ax2, ax3] + B_rf[ax0, ax1, ax2, ax3, vi4]


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


def test_reduction_rfactor_matmul():
    s = tir.Schedule(transformed_matmul, debug_mask="all")
    update = s.get_block("update")
    _, _, _, _, kii = s.get_loops(update)
    rf_block = s.rfactor(kii, 0)
    tvm.ir.assert_structural_equal(s.mod["main"], matmul_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("update_rf")))
    assert s.get(update).same_as(s.get(s.get_block("update")))
    verify_trace_roundtrip(s, mod=transformed_matmul)


def test_reduction_rfactor_square_sum():
    s = tir.Schedule(square_sum, debug_mask="all")
    C = s.get_block("C")
    _, _, j = s.get_loops(C)
    rf_block = s.rfactor(j, 1)
    tvm.ir.assert_structural_equal(s.mod["main"], square_sum_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))
    assert s.get(C).same_as(s.get(s.get_block("C")))
    verify_trace_roundtrip(s, mod=square_sum)


def test_reduction_rfactor_square_sum_square_root():
    s = tir.Schedule(transformed_square_sum_square_root, debug_mask="all")
    C = s.get_block("C")
    _, _, f_i = s.get_loops(C)
    rf_block = s.rfactor(f_i, 0)
    tvm.ir.assert_structural_equal(s.mod["main"], square_sum_square_root_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))
    assert s.get(C).same_as(s.get(s.get_block("C")))
    verify_trace_roundtrip(s, mod=transformed_square_sum_square_root)


def test_reduction_rfactor_square_sum_square_root_factor_one_1():
    s = tir.Schedule(transformed_square_sum_square_root_factor_one_1, debug_mask="all")
    C = s.get_block("C")
    _, _, f_i = s.get_loops(C)
    rf_block = s.rfactor(f_i, 0)
    tvm.ir.assert_structural_equal(s.mod["main"], square_sum_square_root_factor_one_1_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))
    assert s.get(C).same_as(s.get(s.get_block("C")))
    verify_trace_roundtrip(s, mod=transformed_square_sum_square_root_factor_one_1)


def test_reduction_rfactor_square_sum_square_root_factor_one_2():
    s = tir.Schedule(transformed_square_sum_square_root_factor_one_2, debug_mask="all")
    C = s.get_block("C")
    _, f_o, _ = s.get_loops(C)
    rf_block = s.rfactor(f_o, 1)
    tvm.ir.assert_structural_equal(s.mod["main"], square_sum_square_root_factor_one_2_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))
    assert s.get(C).same_as(s.get(s.get_block("C")))
    verify_trace_roundtrip(s, mod=transformed_square_sum_square_root_factor_one_2)


def test_reduction_rfactor_loop_multiple_children():
    s = tir.Schedule(matmul_loop_multiple_children, debug_mask="all")
    k, _, _ = s.get_loops(s.get_block("C"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_stage_pipeline():
    s = tir.Schedule(matmul_not_stage_pipeline, debug_mask="all")
    _, _, k = s.get_loops(s.get_block("C"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_reduction_block1():
    s = tir.Schedule(element_wise, debug_mask="all")
    i, _ = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(i, 0)


def test_reduction_rfactor_not_reduction_block2():
    s = tir.Schedule(rowsum_not_quasi_affine, debug_mask="all")
    _, k = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_reduction_block3():
    s = tir.Schedule(rowsum_not_dominant, debug_mask="all")
    _, k = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_serial_loop():
    s = tir.Schedule(rowsum_not_serial, debug_mask="all")
    _, k = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_same_buffer_access():
    s = tir.Schedule(matmul_not_same_buffer_access, debug_mask="all")
    _, _, k = s.get_loops(s.get_block("C"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_factor_axis_range_fail():
    s = tir.Schedule(transformed_matmul, debug_mask="all")
    _, _, _, _, kii = s.get_loops(s.get_block("update"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(kii, 3)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(kii, -4)


def test_reduction_rfactor_factor_axis_range():
    s = tir.Schedule(transformed_matmul, debug_mask="all")
    update = s.get_block("update")
    _, _, _, _, kii = s.get_loops(update)
    rf_block = s.rfactor(kii, -3)
    tvm.ir.assert_structural_equal(s.mod["main"], matmul_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("update_rf")))
    assert s.get(update).same_as(s.get(s.get_block("update")))
    verify_trace_roundtrip(s, mod=transformed_matmul)


def test_reduction_rfactor_wrong_reduce_pattern1():
    s = tir.Schedule(rowsum_wrong_reduce_pattern1, debug_mask="all")
    _, k = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_wrong_reduce_pattern2():
    s = tir.Schedule(rowsum_wrong_reduce_pattern2, debug_mask="all")
    _, k = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_wrong_loops1():
    s = tir.Schedule(rowsum, debug_mask="all")
    i, _ = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(i, 0)


def test_reduction_rfactor_wrong_loops2():
    s = tir.Schedule(rowsum_transformed, debug_mask="all")
    _, _, k_i = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k_i, 0)


def test_reduction_rfactor_zero_dim():
    s = tir.Schedule(rowsum_zero_dim, debug_mask="all")
    B = s.get_block("B")
    (k,) = s.get_loops(B)
    rf_block = s.rfactor(k, 0)
    tvm.ir.assert_structural_equal(s.mod["main"], rowsum_zero_dim_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("B_rf")))
    assert s.get(B).same_as(s.get(s.get_block("B")))
    verify_trace_roundtrip(s, mod=rowsum_zero_dim)


def test_reduction_rfactor_outermost_loop_multiple_children_fail():  # pylint: disable=invalid-name
    s = tir.Schedule(multiple_reduction_blocks, debug_mask="all")
    _, _, k2o, k2i = s.get_loops(s.get_block("D"))
    _, _, k3o, k3i = s.get_loops(s.get_block("E"))
    _, _, k4o, k4i = s.get_loops(s.get_block("F"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k2o, 0)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k2i, 0)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k3o, 0)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k3i, 0)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k4o, 0)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k4i, 0)


def test_reduction_rfactor_outermost_loop_multiple_children():  # pylint: disable=invalid-name
    s = tir.Schedule(multiple_reduction_blocks, debug_mask="all")
    C = s.get_block("C")
    _, _, k1o, _ = s.get_loops(C)
    rf_block = s.rfactor(k1o, 2)
    tvm.ir.assert_structural_equal(s.mod["main"], multiple_reduction_blocks_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))
    assert s.get(C).same_as(s.get(s.get_block("C")))
    verify_trace_roundtrip(s, mod=multiple_reduction_blocks)


def test_reduction_rfactor_predicate():  # pylint: disable=invalid-name
    s = tir.Schedule(rowsum_predicate, debug_mask="all")
    B = s.get_block("B")
    _, ko, _ = s.get_loops(B)
    # TODO: should be a tvm.tir.ScheduleError
    with pytest.raises(tvm.TVMError):
        rf_block = s.rfactor(ko, 1)


def test_reduction_rfactor_with_annotation():
    s = tir.Schedule(square_sum_with_annotation, debug_mask="all")
    C = s.get_block("C")
    _, _, j = s.get_loops(C)
    rf_block = s.rfactor(j, 1)
    tvm.ir.assert_structural_equal(s.mod["main"], square_sum_with_annotation_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))
    assert s.get(C).same_as(s.get(s.get_block("C")))
    verify_trace_roundtrip(s, mod=square_sum_with_annotation)


def test_reduction_rfactor_spatial_only():
    s = tir.Schedule(rfactor_spatial_only, debug_mask="all")
    block = s.get_block(name="acc", func_name="main")
    _, _, _, _, loop, _ = s.get_loops(block)
    s.rfactor(loop=loop, factor_axis=4)
    tvm.ir.assert_structural_equal(s.mod["main"], rfactor_spatial_only_after)
    verify_trace_roundtrip(s, mod=rfactor_spatial_only)


if __name__ == "__main__":
    tvm.testing.main()
