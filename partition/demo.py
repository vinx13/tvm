"""
How to optimize GEMM on CPU
===========================
**Author**: `Jian Weng <https://github.com/were>`_, \
            `Ruofei Yu <https://github.com/yuruofeifei>`_

(TL;DR) TVM provides abstract interfaces which allows users to depict an algorithm and the
algorithm's implementing organization (the so-called schedule) separately. Typically, writing
algorithm in high-performance schedule breaks the algorithm's readability and modularity. Also,
trying various seemingly promising schedules is time-consuming. With the help of TVM, we can
try these schedules efficiently to enhance the performance.

In this tutorial, we will demonstrate how to use TVM to optimize square matrix multiplication
and achieve 200 times faster than baseline by simply adding 18 extra lines of code.

There are two important optimizations on intense computation applications executed on CPU:
    1. Increase the cache hit rate of memory access. Both complex numerical computation and hot-spot
       memory access can be accelerated from high cache hit rate. This requires us to transform the
       origin memory access pattern to the pattern fits the cache policy.
       transform the data access pattern in the loop body in uniform pattern so that the LLVM
       backend can lower it to SIMD.

Actually, all the methodologies used in this tutorial is a subset of tricks mentioned in this
`repo <https://github.com/flame/how-to-optimize-gemm>`_. Some of them have been applied by TVM
abstraction automatically, but some of them cannot be simply applied due to TVM constraints.

All the experiment results mentioned below, are executed on 2015's 15' MacBook equipped with
Intel i7-4770HQ CPU. The cache line size should be 64 bytes for all the x86 CPUs.
"""

################################################################################################
# Preparation and Baseline
# ------------------------
# In this tutorial, we will demo how to use TVM to optimize matrix multiplication.
# Before actually demonstrating, we first define these variables.
# Then we write a baseline implementation, the simplest way to write a matrix multiplication in TVM.

import tvm
from tvm import te, tir
from tvm.contrib import clang, cblas
import numpy
import timeit

ll_code = clang.create_llvm('gemmMxN__avx2.c', options=['-march=skylake'])

# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M_var = te.var('n')
M = 768
K = 128
N = 288
FLOPS = 2 * M * N * K
# The default tensor type in tvm
dtype = "float32"

# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
target = 'llvm -mcpu=core-avx2'
ctx = tvm.context(target, 0)
print("asdf")
# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), ctx)

REPEAT = 100
np_runing_time = timeit.timeit(setup='import numpy\n'
                                     'M = ' + str(M) + '\n'
                                     'K = ' + str(K) + '\n'
                                     'N = ' + str(N) + '\n'
                                     'dtype = "float32"\n'
                                     'a = numpy.random.rand(M, K).astype(dtype)\n'
                                     'b = numpy.random.rand(K, N).astype(dtype)\n',
                               stmt='answer = numpy.dot(a, b)',
                               number=REPEAT)

print("Numpy running time: %f" % (FLOPS * REPEAT / np_runing_time / 1.0E9))

answer = numpy.dot(a.asnumpy(), b.asnumpy())

# Algorithm
k = te.reduce_axis((0, K), 'k')
A = te.placeholder((M_var, K), name='A')
B = te.placeholder((K, N), name='B')
C = te.compute(
           (M_var, N),
           lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
           name='C')

# Default schedule
s = te.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

bn = 32

# Tensorized
def intrin_gemm(M, N, K):
    assert M == 4
    assert N == 24
    dtype = 'float32'
    A = te.placeholder((K, M), dtype=dtype, name='A')
    B = te.placeholder((K, N), dtype=dtype, name='B')
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((M, N), lambda m, n:
                    te.sum(A[k, m] * B[k, n], axis=[k]), name='C')

    Ab = tvm.tir.decl_buffer(A.shape, A.dtype,
                        name="A",
                        offset_factor=4,
                        strides=[M, 1])
    Bb = tvm.tir.decl_buffer(B.shape, B.dtype,
                        name="B",
                        offset_factor=24,
                        strides=[N, 1])
    Cb = tvm.tir.decl_buffer(C.shape, C.dtype,
                        name="C",
                        offset_factor=1,
                        strides=[te.var('ldc'), 1])

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]
        irb = tvm.tir.ir_builder.create()
        extern_call = tvm.tir.call_extern(
            "int32",
            "sgemm_only_4x24__avx2",
            K,
            irb.buffer_ptr(aa),
            aa.elem_offset,
            irb.buffer_ptr(bb),
            bb.elem_offset,
            irb.buffer_ptr(cc),
            cc.elem_offset,
            cc.strides[0])
        irb.emit(extern_call)
        return irb.get()

    with tvm.target.build_config():
        return te.decl_tensor_intrin(C.op,
                                      intrin_func,
                                      binds={A: Ab, B: Bb, C: Cb})

MTile = 4
NTile = 24

# assert M % MTile == 0
# assert N % NTile == 0

APanel = te.compute(
    (te.indexdiv(M_var, MTile), K, MTile), lambda mtile, k, m: A[m + mtile * MTile, k], name='APanel')
BPanel = te.compute(
    (N / NTile, K, NTile), lambda ntile, k, n: B[k, n + ntile * NTile], name='BPanel')
print("APanel, ", APanel.shape)
print("BPanel, ", BPanel.shape)
k = te.reduce_axis((0, K), name='k')
C = te.compute(
    (M_var, N),
    lambda m, n: te.sum(
        APanel[te.indexdiv(m, MTile), k, te.indexmod(m, MTile)] * BPanel[te.indexdiv(n, NTile), k, te.indexmod(n, NTile)],
        axis=[k]),
    name='C')

print("C", C.shape, M, N)
s = te.create_schedule(C.op)
x, y, z = BPanel.op.axis
# s[BPanel].vectorize(z)
x, y, z = APanel.op.axis
s[APanel].unroll(z)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, NTile)
s[C].reorder(xo, yo, xi, yi)
xii, xiii = s[C].split(xi, factor=MTile)
s[C].pragma(xo, "import_llvm", ll_code)
gemm_intrinsic_function = intrin_gemm(M=MTile, N=NTile, K=K)
print('tensorize')
#s[C].pragma(xiii, 'tensorize', gemm_intrinsic_function)
s[C].pragma(xiii, 'tensorize_hint', True)
#s[C].tensorize(xiii, gemm_intrinsic_function)

print(tvm.lower(s, [A, B, C], simple_mode=False))

auto_tensorize_pass = tir.auto_tensorize([gemm_intrinsic_function])
with tvm.target.build_config(add_lower_pass=[(3, auto_tensorize_pass)], dump_pass_ir=True) as config:
    print(config)
    func = tvm.build(s, [A, B, C], target=target)
print(func.get_source())
assert func
func.save("tensor_gemm.asm")
func.save("tensor_gemm.o")
c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)
func(a, b, c)
print("C shape", c.shape)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=REPEAT)
print('OptTensorize: %f' % (FLOPS / evaluator(a, b, c).mean / 1E9))

