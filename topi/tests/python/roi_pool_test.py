import tvm
import topi
import numpy as np

def test_roi_pool():
    batch_size = 1
    in_channel = 1
    size = 7
    pooled_size = 7
    num_roi = 1
    data_shape = (batch_size, in_channel, size, size)
    rois_shape = (num_roi, 5)
    data=tvm.placeholder(data_shape)
    rois=tvm.placeholder(rois_shape)
    np_data = np.random.randn(*data_shape).reshape(data_shape).astype('float32')
    np_data = np.arange(batch_size*in_channel*size*size).astype('float32').reshape(data_shape)
    np_rois = np.random.randn(*rois_shape).astype('float32')
    np_rois[:, 0] = 0
    #np_rois[0] = [ 0,          1.6889162,   0.02869398, -0.29846725,  1.6345967 ]
    #np_rois[0] = [ 0,          2,   0, 0, 2 ]
    #np_rois[0]=[ 0.     ,    0.2596426,  1.5471624,  1.6333833, -1.0204655]
    #np_rois[0]=[ 0.     ,    0,  2,  2, 0]
    np_rois[0]=[ 0.     ,    -2,  1,  0, 1]

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            out = topi.vision.rcnn.roi_pool(data, rois, pooled_size=pooled_size, spatial_scale=1.0)
            s = topi.generic.schedule_roi_pool(out)

        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_rois = tvm.nd.array(np_rois, ctx)
        tvm_out = tvm.nd.array(np.zeros((num_roi, in_channel, pooled_size, pooled_size)).astype(out.dtype), ctx=ctx)
        print(tvm.lower(s, [data, rois, out], simple_mode=True))
        f = tvm.build(s, [data, rois, out], device)
        f(tvm_data, tvm_rois, tvm_out)

        import mxnet
        mx_ctx = mxnet.gpu(0)
        mx_data = mxnet.nd.array(np_data, mx_ctx)
        mx_rois = mxnet.nd.array(np_rois, mx_ctx)
        mx_out = mxnet.nd.ROIPooling(mx_data, mx_rois, pooled_size=(pooled_size, pooled_size), spatial_scale=1.0)
        mx_out = mx_out.asnumpy()

        tvm_out = tvm_out.asnumpy()

        for i in range(num_roi):
            print(np_rois[i])
            np.testing.assert_allclose(tvm_out[i], mx_out[i], rtol=1e-4)


    for device in ['cuda']:
        check_device(device)


test_roi_pool()
