import numpy as np
import tvm
from tvm import autotvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple

from common import get_all_backend

DEBUG=False
def verify_deformable_conv2d_nchw(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1, deformable_groups=1, add_bias=False, add_relu=False):
    print("Workload: (%d, %d, %d, %d, %d, %d, %d, %d)" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))

    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    out_size = (in_size - (kernel - 1) * dilation - 1 + 2 * padding) // stride + 1
    Offset = tvm.placeholder((batch, deformable_groups * kernel * kernel * 2, out_size, out_size), name='offset')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    bias = tvm.placeholder((num_filter, 1, 1), name='bias')

    a_shape = get_const_tuple(A.shape)
    offset_shape = get_const_tuple(Offset.shape)
    w_shape = get_const_tuple(W.shape)
    bias_shape = get_const_tuple(bias.shape)
    dtype = A.dtype

    #@memoize("topi.tests.test_topi_deformable_conv2d_nchw.verify_deformable_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        offset_np = np.random.randn(*offset_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)

        if DEBUG:
            a_np = np.arange(batch*in_channel*in_height*in_width).reshape(a_shape).astype(dtype)
            offset_np = np.zeros(offset_shape).astype(dtype)
            offset_np[:,0:18,...] = 0.5
            offset_np += 0.5
            w_np = np.ones(w_shape).astype(dtype)

        c_np = topi.testing.deformable_conv2d_nchw_python(a_np, offset_np, w_np, stride, padding,
                                                          dilation, deformable_groups)
        if add_bias:
            b_np = np.random.uniform(size=bias_shape).astype(dtype)
            c_np += b_np
        if add_relu:
            c_np = np.maximum(c_np, 0)
        return a_np, offset_np, w_np, b_np, c_np

    a_np, offset_np, w_np, b_np, c_np = get_ref_data()

    import mxnet as mx
    ctx = mx.gpu(0)
    a_mx = mx.nd.array(a_np).as_in_context(ctx)
    offset_mx = mx.nd.array(offset_np).as_in_context(ctx)
    w_mx = mx.nd.array(w_np).as_in_context(ctx)
    c_mx = mx.nd.contrib.DeformableConvolution(data=a_mx, offset=offset_mx, weight=w_mx, kernel=(kernel,kernel), stride=(stride,stride),pad=(padding,padding), num_filter=num_filter,no_bias=True,bias=None,dilate=(dilation,dilation), num_deformable_group=deformable_groups)
    c_mx = c_mx.asnumpy()
    #print('a_np')
    #print(a_np)
    #print('c_mx')
    #print(c_mx)
    np.testing.assert_allclose(c_np, c_mx,rtol=1e-5)


    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            C = topi.nn.deformable_conv2d(A, Offset, W, stride, padding, dilation, deformable_groups, layout='NCHW',
                                          out_dtype=dtype)
            if add_bias:
                C = topi.add(C, bias)
            if add_relu:
                C = topi.nn.relu(C)
            s = topi.generic.schedule_deformable_conv2d_nchw([C])
            #print(tvm.lower(s, [A, Offset, W, C], simple_mode=True))

            a = tvm.nd.array(a_np, ctx)
            offset = tvm.nd.array(offset_np, ctx)
            w = tvm.nd.array(w_np, ctx)
            b = tvm.nd.array(b_np, ctx)
            c = tvm.nd.array(c_np, ctx)
            if add_bias:
                func = tvm.build(s, [A, Offset, W, bias, C], device)
                func(a, offset, w, b, c)
            else:
                func = tvm.build(s, [A, Offset, W, C], device)
                func(a, offset, w, c)
                print(func.imported_modules[0].get_source())
            tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    for device in ['cuda']:
        check_device(device)

def test_deformable_conv2d_nchw():
    verify_deformable_conv2d_nchw(1, 4, 2, 4, 3, 1, 1, deformable_groups=2)
    verify_deformable_conv2d_nchw(1, 64, 7, 64, 1, 1, 0, deformable_groups=4)
    verify_deformable_conv2d_nchw(1, 1, 3, 1, 1, 1, 0)
    verify_deformable_conv2d_nchw(1, 1, 2, 1, 3, 1, 1)
    verify_deformable_conv2d_nchw(1, 64, 7, 64, 1, 1, 0)
    verify_deformable_conv2d_nchw(1, 64, 7, 64, 3, 1, 1)
    verify_deformable_conv2d_nchw(1, 64, 7, 64, 3, 1, 2, dilation=2)
    verify_deformable_conv2d_nchw(1, 4, 14, 4, 1, 2, 0)

if __name__ == "__main__":
    test_deformable_conv2d_nchw()
