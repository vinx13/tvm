import numpy as np
import logging
import argparse
import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
import tvm
import tvm.relay as relay
import tvm.relay.expr as _expr
from tvm.contrib import graph_runtime
from scipy import stats
import pickle


# Two functions for reading data from record file or raw images
def get_val_data(args,
                 rec_val,
                 batch_size,
                 num_workers=4,
                 shuffle=False):
    rec_val = os.path.expanduser(rec_val)
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    img_size = 299 if args.model == 'inceptionv3' else 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        preprocess_threads  = num_workers,
        shuffle             = shuffle,
        batch_size          = batch_size,
        resize              = 256,
        data_shape          = (3, img_size, img_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return val_data, batch_fn


def calibrate_dataset(args):
    val_data, batch_fn = get_val_data(args, args.rec_val, 1, shuffle=True)
    val_data.reset()
    for i, batch in enumerate(val_data):
        if i > 100:
            break
        data, label = batch_fn(batch, [mx.cpu(0)])
        yield {'data': tvm.nd.array(data[0].asnumpy(), tvm.gpu(0))}


def evaluate(args, graph, lib, params, ctx):
    """Evaluate on the validation set."""
    # setup dataset.
    batch_size = args.batch_size
    val_data, batch_fn = get_val_data(args, args.rec_val, batch_size)
    # create runtime module
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    oshape = (batch_size, args.num_classes)
    out_arr = tvm.nd.empty(oshape, "float32")
    # setup evaluaiton metric
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    val_data.reset()
    acc_top1.reset()
    acc_top5.reset()

    # Execute
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, [mx.cpu(0)])
        m.run(data=data[0].asnumpy())
        m.get_output(0, out_arr)
        acc_top1.update(label, [mx.nd.array(out_arr.asnumpy())])
        acc_top5.update(label, [mx.nd.array(out_arr.asnumpy())])

        if args.log_interval and not (i + 1) % args.log_interval:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            nsamples = (i + 1) * batch_size
            logging.info('[%d samples] validation: acc-top1=%f acc-top5=%f', nsamples, top1, top5)
    logging.info('[final] validation: acc-top1=%f acc-top5=%f', top1, top5)
    with open('record.csv', "a") as f:
        f.write('{}, {}\n'.format(
            args.model, top1))


def build_model(args, gluon_model, original):
    """Build with relay."""
    import tvm
    from tvm import relay
    from tvm.relay import quantize as qtz
    img_size = 299 if args.model == 'inceptionv3' else 224
    data_shape = (args.batch_size, 3, img_size, img_size)
    net, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    target = args.target

    if original:
        # run original model
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(net, target, params=params)
        ctx = tvm.nd.context(target, 0)
        return graph, lib, params, ctx

    with relay.quantize.qconfig(store_lowbit_output=False):
        qgraph = relay.quantize.quantize(net, params=params, dataset=calibrate_dataset(args))

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(qgraph, target)

    if args.save_model is not None:
        prefix = args.save_model
        save_model(prefix, graph, lib, params)

    ctx = tvm.nd.context(target, 0)
    return graph, lib, params, ctx


def save_model(name, graph, lib, params):
    with open(name + '.json', 'w') as f:
        f.write(graph)
    lib.export_library(name + '.so')
    with open(name + '.bin', 'wb') as f:
        f.write(relay.save_param_dict(params))


def main(args):
    gluon_model = vision.get_model(args.model, pretrained=True)
    graph, lib, params, ctx = build_model(args, gluon_model, args.original)
    logging.info("Finish building model %s...", args.model)
    evaluate(args, graph, lib, params, ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ImageNet validation accuracy")
    parser.add_argument("--rec-val", type=str, default="~/.mxnet/datasets/imagenet/rec/val.rec",
                        help="the validation data")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="batch size")
    parser.add_argument("--model", type=str, default="resnet50_v2",
                        help="Name of the model")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="log interval")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--target", type=str, default="cuda",
                        help="target option")
    parser.add_argument("--original", action="store_true",
                        help='whether to use original graph')
    parser.add_argument('--save_model', type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info(args)
    main(args)
