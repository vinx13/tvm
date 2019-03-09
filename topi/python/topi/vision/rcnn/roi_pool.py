from ...util import get_const_tuple
import tvm

@tvm.target.generic_func
def roi_pool(data, rois, pooled_size, spatial_scale):
    _, channel, height, width = get_const_tuple(data.shape)
    num_roi, _ = get_const_tuple(rois.shape)

    def _pool(i, c, ph, pw):
        roi = rois[i]
        batch_index = roi[0].astype('int32')
        roi_start_w, roi_start_h, roi_end_w, roi_end_h = roi[1], roi[2], roi[3], roi[4]

        roi_start_h = tvm.round(roi_start_h * spatial_scale).astype('int32')
        roi_start_w = tvm.round(roi_start_w * spatial_scale).astype('int32')
        roi_end_h = tvm.round(roi_end_h * spatial_scale).astype('int32')
        roi_end_w = tvm.round(roi_end_w * spatial_scale).astype('int32')

        # force malformed ROIs to be 1x1
        roi_h = tvm.max(roi_end_h - roi_start_h + 1, tvm.const(1, 'int32'))
        roi_w = tvm.max(roi_end_w - roi_start_w + 1, tvm.const(1, 'int32'))

        bin_h = roi_h.astype('float32') / tvm.const(pooled_size, 'float32')
        bin_w = roi_w.astype('float32') / tvm.const(pooled_size, 'float32')

        hstart = tvm.floor(ph * bin_h).astype('int32')
        wstart = tvm.floor(pw * bin_w).astype('int32')
        hend = tvm.ceil((ph + 1) * bin_h).astype('int32')
        wend = tvm.ceil((pw + 1) * bin_w).astype('int32')
        hstart = tvm.make._OpMin(tvm.make._OpMax(hstart + roi_start_h, 0), height)
        wstart = tvm.make._OpMin(tvm.make._OpMax(wstart + roi_start_w, 0), width)
        hend = tvm.make._OpMin(tvm.make._OpMax(hend + roi_start_h, 0), height)
        wend = tvm.make._OpMin(tvm.make._OpMax(wend + roi_start_w, 0), width)
        #return wend.astype('float32')

        non_empty = tvm.all(hstart < hend, wstart < wend)
        min_value = lambda dtype: tvm.expr.Select(non_empty, tvm.min_value(dtype),
                                             tvm.const(0.0, dtype))
        _max = tvm.comm_reducer(lambda x, y: tvm.make._OpMax(x, y), min_value, name='max')
        rh = tvm.reduce_axis((0, hend - hstart), 'rh')
        rw = tvm.reduce_axis((0, wend - wstart), 'rw')
        #return hstart.astype('float32')
        return _max(data[batch_index, c, hstart+rh, wstart+rw], axis=[rh,rw])

    return tvm.compute((num_roi, channel, pooled_size, pooled_size), _pool, tag="roi_pool")
