from . import ir_pass

def auto_tensorize(tensor_intrins):
    def f(stmt, s):
        return ir_pass.AutoTensorize(stmt, s, tensor_intrins)
    return f