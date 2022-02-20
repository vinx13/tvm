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
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks


def packed_index_map_func(m, n):
    return m // 16, n // 16, m % 16, n % 16


@T.prim_func
def two_elementwise(A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]) -> None:
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def two_elementwise_transformed_intermediate_buffer(
    A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]
) -> None:
    B = T.alloc_buffer((8, 8, 16, 16), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi // 16, vj // 16, vi % 16, vj % 16] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi // 16, vj // 16, vi % 16, vj % 16] + 1.0


@T.prim_func
def two_elementwise_transformed_input_buffer(
    A: T.Buffer[(8, 8, 16, 16), "float32"], C: T.Buffer[(128, 128), "float32"]
) -> None:
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi // 16, vj // 16, vi % 16, vj % 16] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def two_elementwise_transformed_output_buffer(
    A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(8, 8, 16, 16), "float32"]
) -> None:
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi // 16, vj // 16, vi % 16, vj % 16] = B[vi, vj] + 1.0


@T.prim_func
def Conv2D(var_inputs: T.handle, var_weight: T.handle, var_conv2d_nhwc: T.handle) -> None:
    inputs = T.match_buffer(var_inputs, [1, 224, 224, 3], align=128, offset_factor=1)
    weight = T.match_buffer(var_weight, [7, 7, 3, 64], align=128, offset_factor=1)
    conv2d_nhwc = T.match_buffer(var_conv2d_nhwc, [1, 112, 112, 64], align=128, offset_factor=1)
    PadInput = T.alloc_buffer([1, 230, 230, 3], elem_offset=0, align=128, offset_factor=1)
    for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
        with T.block("PadInput"):
            i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(((((i1_1 >= 3) and (i1_1 < 227)) and (i2_1 >= 3)) and (i2_1 < 227)), inputs[i0_1, (i1_1 - 3), (i2_1 - 3), i3_1], T.float32(0), dtype="float32")
    for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 112, 112, 64, 7, 7, 3):
        with T.block("conv2d_nhwc"):
            n, h, w, co, rh, rw, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
            with T.init():
                conv2d_nhwc[n, h, w, co] = T.float32(0)
            conv2d_nhwc[n, h, w, co] = (conv2d_nhwc[n, h, w, co] + (PadInput[n, ((h*2) + rh), ((w*2) + rw), ((T.floordiv(co, 64)*3) + rc)]*weight[rh, rw, rc, co]))


@T.prim_func
def Conv2D_transformed(var_inputs: T.handle, var_weight: T.handle, var_conv2d_nhwc: T.handle) -> None:
    inputs = T.match_buffer(var_inputs, [1, 224, 224, 3], dtype="float32", offset_factor=1)
    weight = T.match_buffer(var_weight, [7, 7, 3, 64], dtype="float32", offset_factor=1)
    conv2d_nhwc = T.match_buffer(var_conv2d_nhwc, [1, 112, 112, 64], dtype="float32", offset_factor=1)
    # body
    # with T.block("root")
    PadInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
    for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
        with T.block("PadInput"):
            i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(inputs[i0_1, i1_1 - 3, i2_1 - 3, i3_1])
            T.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
            PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(i1_1 >= 3 and i1_1 < 227 and i2_1 >= 3 and i2_1 < 227, inputs[i0_1, i1_1 - 3, i2_1 - 3, i3_1], T.float32(0), dtype="float32")
    for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 112, 112, 64, 7, 7, 3):
        with T.block("conv2d_nhwc"):
            bv = T.axis.spatial(12544, i0 * 12544 + i1 * 112 + i2)
            bv_1 = T.axis.spatial(64, i3)
            bv_2 = T.axis.spatial(147, i4 * 21 + i5 * 3 + i6)
            T.reads(conv2d_nhwc[0, bv // 112 % 112, bv % 112, bv_1], PadInput[0, bv // 112 % 112 * 2 + bv_2 // 21 % 7, bv % 112 * 2 + bv_2 // 3 % 7, bv_1 // 64 * 3 + bv_2 % 3], weight[bv_2 // 21 % 7, bv_2 // 3 % 7, bv_2 % 3, bv_1])
            T.writes(conv2d_nhwc[0, bv // 112 % 112, bv % 112, bv_1])
            with T.init():
                conv2d_nhwc[0, bv // 112 % 112, bv % 112, bv_1] = T.float32(0)
            conv2d_nhwc[0, bv // 112 % 112, bv % 112, bv_1] = conv2d_nhwc[0, bv // 112 % 112, bv % 112, bv_1] + PadInput[0, bv // 112 % 112 * 2 + bv_2 // 21 % 7, bv % 112 * 2 + bv_2 // 3 % 7, bv_1 // 64 * 3 + bv_2 % 3] * weight[bv_2 // 21 % 7, bv_2 // 3 % 7, bv_2 % 3, bv_1]


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks
# fmt: on


def test_two_elementwise_transform_intermediate_buffer():
    sch = tir.Schedule(two_elementwise, debug_mask="all")
    block = sch.get_block("B")
    sch.transform_layout(block, 0, "write", lambda m, n: (m // 16, n // 16, m % 16, n % 16))
    tvm.ir.assert_structural_equal(two_elementwise_transformed_intermediate_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_two_elementwise_transform_input_buffer():
    sch = tir.Schedule(two_elementwise, debug_mask="all")
    block = sch.get_block("B")
    sch.transform_layout(block, 0, "read", packed_index_map_func)
    tvm.ir.assert_structural_equal(two_elementwise_transformed_input_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_two_elementwise_transform_output_buffer():
    sch = tir.Schedule(two_elementwise, debug_mask="all")
    block = sch.get_block("C")
    sch.transform_layout(block, 0, "write", packed_index_map_func)
    tvm.ir.assert_structural_equal(two_elementwise_transformed_output_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_block_rewrite_layout():
    print(Conv2D.script())
    sch = tir.Schedule(Conv2D, debug_mask="all")
    block = sch.get_block("conv2d_nhwc")
    sch.transform_block_layout(block, lambda n, h, w, co, rh, rw, rc: (n * 112 * 112 + h * 112 + w, co, rh * 7 * 3 + rw * 3 + rc))
    print(sch.mod.script())


if __name__ == "__main__":
    # sys.exit(pytest.main([__file__] + sys.argv[1:]))
    test_block_rewrite_layout()
