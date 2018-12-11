def test_fuse_conv2d():
    def elu(data):
        return -0.5 * sym.relu(1 - sym.exp(data)) + sym.relu(data)

    def get_sym(out_channel):
        data = sym.Variable(name="data")
        conv1 = sym.conv2d(data=data, kernel_size=(1, 1), channels=out_channel, padding=0,
                          layout="NCHW", kernel_layout="OIHW", use_bias=False)
        conv2 = sym.conv2d(data=data, kernel_size=(1, 1), channels=out_channel, padding=0,
                          layout="NCHW", kernel_layout="OIHW", use_bias=False)
        output = sym.concatenate([conv1, conv2])
        return output


if __name__ == '__main__':
    test_fuse_conv2