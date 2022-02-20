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
import tvm
from tvm import tir, te
from tvm.script import tir as T


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.ApplyBlockBoundPredicate()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


def _check_print(original):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.LowerCrossThreadReduction()(mod)
    mod = tvm.tir.transform.LowerInitBlock()(mod)
    mod = tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    print(mod["main"].script())
    mod = tvm.tir.transform.ApplyBlockBoundPredicate()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    print(mod["main"].script())


# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks


@T.prim_func
def read_out_of_bound_after_compute_at(A: T.Buffer[(16,), "float32"], C: T.Buffer[(16,), "float32"]) -> None:
    B = T.alloc_buffer([16], dtype="float32")
    for j in T.serial(16):
        for ax0 in T.serial(2):
            with T.block("B"):
                v = T.axis.spatial(16, j + ax0)
                T.reads(A[v])
                T.writes(B[v])
                T.block_attr({"require_bound_predicate":v >= 0 and v < 16})
                B[v] = A[v]
        with T.block("C"):
            v = T.axis.spatial(16, j)
            T.reads(B[v : v + 2])
            T.writes(C[v])
            C[v] = T.if_then_else(v < 15, T.max(B[v], B[v + 1]), B[v], dtype="float32")


@T.prim_func
def tiled_pooling_cache_after_compute_at(a: T.handle, b: T.handle) -> None:
    X = T.match_buffer(a, [224, 224], dtype="float32")
    Y = T.match_buffer(b, [224, 224], dtype="float32")
    cache = T.alloc_buffer([224, 224], dtype="float32")
    dache = T.alloc_buffer([224, 224], dtype="float32")
    for hh_0, ww_0 in T.grid(28, 28):
        for ax0, ax1 in T.grid(10, 10):
            with T.block("cache"):
                h = T.axis.spatial(224, hh_0 * 8 + ax0 - 1)
                w = T.axis.spatial(224, ww_0 * 8 + ax1 - 1)
                T.reads(X[h, w])
                T.writes(cache[h, w])
                T.block_attr({"require_bound_predicate":h >= 0 and h < 224 and w >= 0 and w < 224})
                cache[h, w] = X[h, w]
        for ax0, ax1 in T.grid(10, 10):
            with T.block("dache"):
                h = T.axis.spatial(224, hh_0 * 8 + ax0 - 1)
                w = T.axis.spatial(224, ww_0 * 8 + ax1 - 1)
                T.reads(X[h, w])
                T.writes(dache[h, w])
                T.block_attr({"require_bound_predicate":h >= 0 and h < 224 and w >= 0 and w < 224})
                dache[h, w] = X[h, w]
        for hh_1, ww_1, khh, kww in T.grid(8, 8, 3, 3):
            with T.block("compute"):
                h = T.axis.spatial(224, hh_0 * 8 + hh_1)
                w = T.axis.spatial(224, ww_0 * 8 + ww_1)
                kh, kw = T.axis.remap("RR", [khh, kww])
                T.reads([Y[h, w], cache[h + kh - 1, w + kw - 1], dache[h + kh - 1, w + kw - 1]])
                T.writes([Y[h, w]])
                with T.init():
                    Y[h, w] = 0.0
                Y[h, w] = T.max(Y[h, w], T.if_then_else(
                    T.likely(1 <= h + kh, dtype="bool") and \
                    T.likely(h + kh < 225, dtype="bool") and \
                    T.likely(1 <= w + kw, dtype="bool") and \
                    T.likely(w + kw < 225, dtype="bool"),
                    cache[h + kh - 1, w + kw - 1]+ dache[h + kh - 1, w + kw - 1], 0.0, dtype="float32"))


@T.prim_func
def batch_norm_after_compute_at(A: T.Buffer[(1, 256, 256), "float32"], D: T.Buffer[(1,), "float32"]) -> None:
    for i0_0 in T.serial(1):
        with T.block():
            T.reads(A[0 : 64, 0 : 256, 0 : 256])
            T.writes(D[0 : 64])
            C = T.alloc_buffer([1], dtype="float32")
            for ax0, ax1, ax2 in T.grid(64, 256, 256):
                with T.block("C"):
                    b = T.axis.spatial(1, ax0)
                    i, j = T.axis.remap("RR", [ax1, ax2])
                    T.reads(C[b], A[b, i, j])
                    T.writes(C[b])
                    T.block_attr({"require_bound_predicate":b >= 0 and b < 1})
                    if i == 0 and j == 0:
                        C[b] = T.float32(0)
                    C[b] = C[b] + A[b, i, j] * A[b, i, j]
            for i0_1 in T.thread_binding(64, thread="threadIdx.x"):
                with T.block("D"):
                    b = T.axis.spatial(1, i0_1)
                    T.where(i0_1 < 1)
                    T.reads(C[b])
                    T.writes(D[b])
                    D[b] = T.sqrt(C[b], dtype="float32")


@T.prim_func
def transformed_batch_norm(A: T.Buffer[(1, 256, 256), "float32"], D: T.Buffer[(1,), "float32"]) -> None:
    for i0_0 in T.serial(1):
        with T.block():
            T.reads(A[0 : 64, 0 : 256, 0 : 256])
            T.writes(D[0 : 64])
            C = T.alloc_buffer([1], dtype="float32")
            for ax0, ax1, ax2 in T.grid(1, 256, 256):
                with T.block("C"):
                    b = T.axis.spatial(1, 0)
                    i, j = T.axis.remap("RR", [ax1, ax2])
                    T.reads(C[b], A[b, i, j])
                    T.writes(C[b])
                    if i == 0 and j == 0:
                        C[b] = T.float32(0)
                    C[b] = C[b] + A[b, i, j] * A[b, i, j]
            for i0_1 in T.thread_binding(64, thread="threadIdx.x"):
                with T.block("D"):
                    b = T.axis.spatial(1, i0_1)
                    T.where(i0_1 < 1)
                    T.reads(C[b])
                    T.writes(D[b])
                    D[b] = T.sqrt(C[b], dtype="float32")


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks
# fmt: on


def test_read_out_of_bound():
    # This IR should not be mutated in this pass.
    _check(read_out_of_bound_after_compute_at, read_out_of_bound_after_compute_at)


def test_tiled_pooling_cache():
    # This IR should not be mutated in this pass.
    _check(tiled_pooling_cache_after_compute_at, tiled_pooling_cache_after_compute_at)


def test_batch_norm():
    _check(batch_norm_after_compute_at, transformed_batch_norm)


def test_lower_te():
    x = te.placeholder((1,))
    y = te.compute((1,), lambda i: x[i] + 2)
    s = te.create_schedule(y.op)
    orig_mod = tvm.driver.build_module.schedule_to_module(s, [x, y])
    mod = tvm.tir.transform.ApplyBlockBoundPredicate()(orig_mod)
    tvm.ir.assert_structural_equal(mod, orig_mod)  # FlattenBuffer should do nothing on TE


if __name__ == "__main__":
    test_read_out_of_bound()
    test_tiled_pooling_cache()
    test_batch_norm()
    test_lower_te()
