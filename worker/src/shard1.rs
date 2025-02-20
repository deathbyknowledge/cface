// Generated from ONNX "pytorch/shard1.onnx" by burn-import
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::nn::PaddingConfig2d;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    conv2d19: Conv2d<B>,
    conv2d20: Conv2d<B>,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    conv2d23: Conv2d<B>,
    conv2d24: Conv2d<B>,
    conv2d25: Conv2d<B>,
    conv2d26: Conv2d<B>,
    conv2d27: Conv2d<B>,
    conv2d28: Conv2d<B>,
    conv2d29: Conv2d<B>,
    conv2d30: Conv2d<B>,
    conv2d31: Conv2d<B>,
    conv2d32: Conv2d<B>,
    conv2d33: Conv2d<B>,
    conv2d34: Conv2d<B>,
    conv2d35: Conv2d<B>,
    conv2d36: Conv2d<B>,
    conv2d37: Conv2d<B>,
    conv2d38: Conv2d<B>,
    conv2d39: Conv2d<B>,
    conv2d40: Conv2d<B>,
    conv2d41: Conv2d<B>,
    conv2d42: Conv2d<B>,
    conv2d43: Conv2d<B>,
    conv2d44: Conv2d<B>,
    conv2d45: Conv2d<B>,
    maxpool2d2: MaxPool2d,
    conv2d46: Conv2d<B>,
    conv2d47: Conv2d<B>,
    conv2d48: Conv2d<B>,
    conv2d49: Conv2d<B>,
    conv2d50: Conv2d<B>,
    conv2d51: Conv2d<B>,
    conv2d52: Conv2d<B>,
    conv2d53: Conv2d<B>,
    conv2d54: Conv2d<B>,
    conv2d55: Conv2d<B>,
    conv2d56: Conv2d<B>,
    conv2d57: Conv2d<B>,
    conv2d58: Conv2d<B>,
    conv2d59: Conv2d<B>,
    conv2d60: Conv2d<B>,
    conv2d61: Conv2d<B>,
    conv2d62: Conv2d<B>,
    conv2d63: Conv2d<B>,
    conv2d64: Conv2d<B>,
    conv2d65: Conv2d<B>,
    conv2d66: Conv2d<B>,
    conv2d67: Conv2d<B>,
    conv2d68: Conv2d<B>,
    conv2d69: Conv2d<B>,
    conv2d70: Conv2d<B>,
    conv2d71: Conv2d<B>,
    conv2d72: Conv2d<B>,
    conv2d73: Conv2d<B>,
    conv2d74: Conv2d<B>,
    conv2d75: Conv2d<B>,
    conv2d76: Conv2d<B>,
    conv2d77: Conv2d<B>,
    conv2d78: Conv2d<B>,
    conv2d79: Conv2d<B>,
    conv2d80: Conv2d<B>,
    conv2d81: Conv2d<B>,
    conv2d82: Conv2d<B>,
    conv2d83: Conv2d<B>,
    conv2d84: Conv2d<B>,
    conv2d85: Conv2d<B>,
    conv2d86: Conv2d<B>,
    conv2d87: Conv2d<B>,
    conv2d88: Conv2d<B>,
    conv2d89: Conv2d<B>,
    conv2d90: Conv2d<B>,
    conv2d91: Conv2d<B>,
    conv2d92: Conv2d<B>,
    conv2d93: Conv2d<B>,
    conv2d94: Conv2d<B>,
    conv2d95: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}


impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d4 = Conv2dConfig::new([64, 80], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([80, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([192, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([96, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([96, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([96, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([96, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([256, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([96, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([256, 384], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([256, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([192, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([192, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d2 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d46 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([128, 128], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([128, 128], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([256, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([128, 128], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([128, 128], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([256, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d56 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d58 = Conv2dConfig::new([128, 128], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d59 = Conv2dConfig::new([128, 128], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d60 = Conv2dConfig::new([256, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d62 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d63 = Conv2dConfig::new([128, 128], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d64 = Conv2dConfig::new([128, 128], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([256, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d66 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d68 = Conv2dConfig::new([128, 128], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d69 = Conv2dConfig::new([128, 128], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d70 = Conv2dConfig::new([256, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d71 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d72 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d73 = Conv2dConfig::new([128, 128], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d74 = Conv2dConfig::new([128, 128], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d75 = Conv2dConfig::new([256, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d76 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d77 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d78 = Conv2dConfig::new([128, 128], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d79 = Conv2dConfig::new([128, 128], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d80 = Conv2dConfig::new([256, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d81 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d82 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d83 = Conv2dConfig::new([128, 128], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d84 = Conv2dConfig::new([128, 128], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d85 = Conv2dConfig::new([256, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d86 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d87 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d88 = Conv2dConfig::new([128, 128], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d89 = Conv2dConfig::new([128, 128], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d90 = Conv2dConfig::new([256, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d91 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d92 = Conv2dConfig::new([896, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d93 = Conv2dConfig::new([128, 128], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d94 = Conv2dConfig::new([128, 128], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d95 = Conv2dConfig::new([256, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            maxpool2d1,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            conv2d37,
            conv2d38,
            conv2d39,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            conv2d44,
            conv2d45,
            maxpool2d2,
            conv2d46,
            conv2d47,
            conv2d48,
            conv2d49,
            conv2d50,
            conv2d51,
            conv2d52,
            conv2d53,
            conv2d54,
            conv2d55,
            conv2d56,
            conv2d57,
            conv2d58,
            conv2d59,
            conv2d60,
            conv2d61,
            conv2d62,
            conv2d63,
            conv2d64,
            conv2d65,
            conv2d66,
            conv2d67,
            conv2d68,
            conv2d69,
            conv2d70,
            conv2d71,
            conv2d72,
            conv2d73,
            conv2d74,
            conv2d75,
            conv2d76,
            conv2d77,
            conv2d78,
            conv2d79,
            conv2d80,
            conv2d81,
            conv2d82,
            conv2d83,
            conv2d84,
            conv2d85,
            conv2d86,
            conv2d87,
            conv2d88,
            conv2d89,
            conv2d90,
            conv2d91,
            conv2d92,
            conv2d93,
            conv2d94,
            conv2d95,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input1: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d1_out1 = self.conv2d1.forward(input1);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu3_out1);
        let conv2d4_out1 = self.conv2d4.forward(maxpool2d1_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu4_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu6_out1.clone());
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu6_out1.clone());
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = self.conv2d9.forward(relu8_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu6_out1.clone());
        let relu10_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu10_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu11_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d12_out1);
        let concat1_out1 =
            burn::tensor::Tensor::cat([relu7_out1, relu9_out1, relu12_out1].into(), 1);
        let conv2d13_out1 = self.conv2d13.forward(concat1_out1);
        let constant1_out1: f32 = 0.17f32;
        let mul1_out1 = conv2d13_out1.mul_scalar(constant1_out1);
        let add1_out1 = mul1_out1.add(relu6_out1);
        let relu13_out1 = burn::tensor::activation::relu(add1_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu13_out1.clone());
        let relu14_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu13_out1.clone());
        let relu15_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let conv2d16_out1 = self.conv2d16.forward(relu15_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu13_out1.clone());
        let relu17_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu17_out1);
        let relu18_out1 = burn::tensor::activation::relu(conv2d18_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu18_out1);
        let relu19_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let concat2_out1 =
            burn::tensor::Tensor::cat([relu14_out1, relu16_out1, relu19_out1].into(), 1);
        let conv2d20_out1 = self.conv2d20.forward(concat2_out1);
        let constant2_out1: f32 = 0.17f32;
        let mul2_out1 = conv2d20_out1.mul_scalar(constant2_out1);
        let add2_out1 = mul2_out1.add(relu13_out1);
        let relu20_out1 = burn::tensor::activation::relu(add2_out1);
        let conv2d21_out1 = self.conv2d21.forward(relu20_out1.clone());
        let relu21_out1 = burn::tensor::activation::relu(conv2d21_out1);
        let conv2d22_out1 = self.conv2d22.forward(relu20_out1.clone());
        let relu22_out1 = burn::tensor::activation::relu(conv2d22_out1);
        let conv2d23_out1 = self.conv2d23.forward(relu22_out1);
        let relu23_out1 = burn::tensor::activation::relu(conv2d23_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu20_out1.clone());
        let relu24_out1 = burn::tensor::activation::relu(conv2d24_out1);
        let conv2d25_out1 = self.conv2d25.forward(relu24_out1);
        let relu25_out1 = burn::tensor::activation::relu(conv2d25_out1);
        let conv2d26_out1 = self.conv2d26.forward(relu25_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv2d26_out1);
        let concat3_out1 =
            burn::tensor::Tensor::cat([relu21_out1, relu23_out1, relu26_out1].into(), 1);
        let conv2d27_out1 = self.conv2d27.forward(concat3_out1);
        let constant3_out1: f32 = 0.17f32;
        let mul3_out1 = conv2d27_out1.mul_scalar(constant3_out1);
        let add3_out1 = mul3_out1.add(relu20_out1);
        let relu27_out1 = burn::tensor::activation::relu(add3_out1);
        let conv2d28_out1 = self.conv2d28.forward(relu27_out1.clone());
        let relu28_out1 = burn::tensor::activation::relu(conv2d28_out1);
        let conv2d29_out1 = self.conv2d29.forward(relu27_out1.clone());
        let relu29_out1 = burn::tensor::activation::relu(conv2d29_out1);
        let conv2d30_out1 = self.conv2d30.forward(relu29_out1);
        let relu30_out1 = burn::tensor::activation::relu(conv2d30_out1);
        let conv2d31_out1 = self.conv2d31.forward(relu27_out1.clone());
        let relu31_out1 = burn::tensor::activation::relu(conv2d31_out1);
        let conv2d32_out1 = self.conv2d32.forward(relu31_out1);
        let relu32_out1 = burn::tensor::activation::relu(conv2d32_out1);
        let conv2d33_out1 = self.conv2d33.forward(relu32_out1);
        let relu33_out1 = burn::tensor::activation::relu(conv2d33_out1);
        let concat4_out1 =
            burn::tensor::Tensor::cat([relu28_out1, relu30_out1, relu33_out1].into(), 1);
        let conv2d34_out1 = self.conv2d34.forward(concat4_out1);
        let constant4_out1: f32 = 0.17f32;
        let mul4_out1 = conv2d34_out1.mul_scalar(constant4_out1);
        let add4_out1 = mul4_out1.add(relu27_out1);
        let relu34_out1 = burn::tensor::activation::relu(add4_out1);
        let conv2d35_out1 = self.conv2d35.forward(relu34_out1.clone());
        let relu35_out1 = burn::tensor::activation::relu(conv2d35_out1);
        let conv2d36_out1 = self.conv2d36.forward(relu34_out1.clone());
        let relu36_out1 = burn::tensor::activation::relu(conv2d36_out1);
        let conv2d37_out1 = self.conv2d37.forward(relu36_out1);
        let relu37_out1 = burn::tensor::activation::relu(conv2d37_out1);
        let conv2d38_out1 = self.conv2d38.forward(relu34_out1.clone());
        let relu38_out1 = burn::tensor::activation::relu(conv2d38_out1);
        let conv2d39_out1 = self.conv2d39.forward(relu38_out1);
        let relu39_out1 = burn::tensor::activation::relu(conv2d39_out1);
        let conv2d40_out1 = self.conv2d40.forward(relu39_out1);
        let relu40_out1 = burn::tensor::activation::relu(conv2d40_out1);
        let concat5_out1 =
            burn::tensor::Tensor::cat([relu35_out1, relu37_out1, relu40_out1].into(), 1);
        let conv2d41_out1 = self.conv2d41.forward(concat5_out1);
        let constant5_out1: f32 = 0.17f32;
        let mul5_out1 = conv2d41_out1.mul_scalar(constant5_out1);
        let add5_out1 = mul5_out1.add(relu34_out1);
        let relu41_out1 = burn::tensor::activation::relu(add5_out1);
        let conv2d42_out1 = self.conv2d42.forward(relu41_out1.clone());
        let relu42_out1 = burn::tensor::activation::relu(conv2d42_out1);
        let conv2d43_out1 = self.conv2d43.forward(relu41_out1.clone());
        let relu43_out1 = burn::tensor::activation::relu(conv2d43_out1);
        let conv2d44_out1 = self.conv2d44.forward(relu43_out1);
        let relu44_out1 = burn::tensor::activation::relu(conv2d44_out1);
        let conv2d45_out1 = self.conv2d45.forward(relu44_out1);
        let relu45_out1 = burn::tensor::activation::relu(conv2d45_out1);
        let maxpool2d2_out1 = self.maxpool2d2.forward(relu41_out1);
        let concat6_out1 =
            burn::tensor::Tensor::cat([relu42_out1, relu45_out1, maxpool2d2_out1].into(), 1);
        let conv2d46_out1 = self.conv2d46.forward(concat6_out1.clone());
        let relu46_out1 = burn::tensor::activation::relu(conv2d46_out1);
        let conv2d47_out1 = self.conv2d47.forward(concat6_out1.clone());
        let relu47_out1 = burn::tensor::activation::relu(conv2d47_out1);
        let conv2d48_out1 = self.conv2d48.forward(relu47_out1);
        let relu48_out1 = burn::tensor::activation::relu(conv2d48_out1);
        let conv2d49_out1 = self.conv2d49.forward(relu48_out1);
        let relu49_out1 = burn::tensor::activation::relu(conv2d49_out1);
        let concat7_out1 = burn::tensor::Tensor::cat([relu46_out1, relu49_out1].into(), 1);
        let conv2d50_out1 = self.conv2d50.forward(concat7_out1);
        let constant6_out1: f32 = 0.1f32;
        let mul6_out1 = conv2d50_out1.mul_scalar(constant6_out1);
        let add6_out1 = mul6_out1.add(concat6_out1);
        let relu50_out1 = burn::tensor::activation::relu(add6_out1);
        let conv2d51_out1 = self.conv2d51.forward(relu50_out1.clone());
        let relu51_out1 = burn::tensor::activation::relu(conv2d51_out1);
        let conv2d52_out1 = self.conv2d52.forward(relu50_out1.clone());
        let relu52_out1 = burn::tensor::activation::relu(conv2d52_out1);
        let conv2d53_out1 = self.conv2d53.forward(relu52_out1);
        let relu53_out1 = burn::tensor::activation::relu(conv2d53_out1);
        let conv2d54_out1 = self.conv2d54.forward(relu53_out1);
        let relu54_out1 = burn::tensor::activation::relu(conv2d54_out1);
        let concat8_out1 = burn::tensor::Tensor::cat([relu51_out1, relu54_out1].into(), 1);
        let conv2d55_out1 = self.conv2d55.forward(concat8_out1);
        let constant7_out1: f32 = 0.1f32;
        let mul7_out1 = conv2d55_out1.mul_scalar(constant7_out1);
        let add7_out1 = mul7_out1.add(relu50_out1);
        let relu55_out1 = burn::tensor::activation::relu(add7_out1);
        let conv2d56_out1 = self.conv2d56.forward(relu55_out1.clone());
        let relu56_out1 = burn::tensor::activation::relu(conv2d56_out1);
        let conv2d57_out1 = self.conv2d57.forward(relu55_out1.clone());
        let relu57_out1 = burn::tensor::activation::relu(conv2d57_out1);
        let conv2d58_out1 = self.conv2d58.forward(relu57_out1);
        let relu58_out1 = burn::tensor::activation::relu(conv2d58_out1);
        let conv2d59_out1 = self.conv2d59.forward(relu58_out1);
        let relu59_out1 = burn::tensor::activation::relu(conv2d59_out1);
        let concat9_out1 = burn::tensor::Tensor::cat([relu56_out1, relu59_out1].into(), 1);
        let conv2d60_out1 = self.conv2d60.forward(concat9_out1);
        let constant8_out1: f32 = 0.1f32;
        let mul8_out1 = conv2d60_out1.mul_scalar(constant8_out1);
        let add8_out1 = mul8_out1.add(relu55_out1);
        let relu60_out1 = burn::tensor::activation::relu(add8_out1);
        let conv2d61_out1 = self.conv2d61.forward(relu60_out1.clone());
        let relu61_out1 = burn::tensor::activation::relu(conv2d61_out1);
        let conv2d62_out1 = self.conv2d62.forward(relu60_out1.clone());
        let relu62_out1 = burn::tensor::activation::relu(conv2d62_out1);
        let conv2d63_out1 = self.conv2d63.forward(relu62_out1);
        let relu63_out1 = burn::tensor::activation::relu(conv2d63_out1);
        let conv2d64_out1 = self.conv2d64.forward(relu63_out1);
        let relu64_out1 = burn::tensor::activation::relu(conv2d64_out1);
        let concat10_out1 = burn::tensor::Tensor::cat([relu61_out1, relu64_out1].into(), 1);
        let conv2d65_out1 = self.conv2d65.forward(concat10_out1);
        let constant9_out1: f32 = 0.1f32;
        let mul9_out1 = conv2d65_out1.mul_scalar(constant9_out1);
        let add9_out1 = mul9_out1.add(relu60_out1);
        let relu65_out1 = burn::tensor::activation::relu(add9_out1);
        let conv2d66_out1 = self.conv2d66.forward(relu65_out1.clone());
        let relu66_out1 = burn::tensor::activation::relu(conv2d66_out1);
        let conv2d67_out1 = self.conv2d67.forward(relu65_out1.clone());
        let relu67_out1 = burn::tensor::activation::relu(conv2d67_out1);
        let conv2d68_out1 = self.conv2d68.forward(relu67_out1);
        let relu68_out1 = burn::tensor::activation::relu(conv2d68_out1);
        let conv2d69_out1 = self.conv2d69.forward(relu68_out1);
        let relu69_out1 = burn::tensor::activation::relu(conv2d69_out1);
        let concat11_out1 = burn::tensor::Tensor::cat([relu66_out1, relu69_out1].into(), 1);
        let conv2d70_out1 = self.conv2d70.forward(concat11_out1);
        let constant10_out1: f32 = 0.1f32;
        let mul10_out1 = conv2d70_out1.mul_scalar(constant10_out1);
        let add10_out1 = mul10_out1.add(relu65_out1);
        let relu70_out1 = burn::tensor::activation::relu(add10_out1);
        let conv2d71_out1 = self.conv2d71.forward(relu70_out1.clone());
        let relu71_out1 = burn::tensor::activation::relu(conv2d71_out1);
        let conv2d72_out1 = self.conv2d72.forward(relu70_out1.clone());
        let relu72_out1 = burn::tensor::activation::relu(conv2d72_out1);
        let conv2d73_out1 = self.conv2d73.forward(relu72_out1);
        let relu73_out1 = burn::tensor::activation::relu(conv2d73_out1);
        let conv2d74_out1 = self.conv2d74.forward(relu73_out1);
        let relu74_out1 = burn::tensor::activation::relu(conv2d74_out1);
        let concat12_out1 = burn::tensor::Tensor::cat([relu71_out1, relu74_out1].into(), 1);
        let conv2d75_out1 = self.conv2d75.forward(concat12_out1);
        let constant11_out1: f32 = 0.1f32;
        let mul11_out1 = conv2d75_out1.mul_scalar(constant11_out1);
        let add11_out1 = mul11_out1.add(relu70_out1);
        let relu75_out1 = burn::tensor::activation::relu(add11_out1);
        let conv2d76_out1 = self.conv2d76.forward(relu75_out1.clone());
        let relu76_out1 = burn::tensor::activation::relu(conv2d76_out1);
        let conv2d77_out1 = self.conv2d77.forward(relu75_out1.clone());
        let relu77_out1 = burn::tensor::activation::relu(conv2d77_out1);
        let conv2d78_out1 = self.conv2d78.forward(relu77_out1);
        let relu78_out1 = burn::tensor::activation::relu(conv2d78_out1);
        let conv2d79_out1 = self.conv2d79.forward(relu78_out1);
        let relu79_out1 = burn::tensor::activation::relu(conv2d79_out1);
        let concat13_out1 = burn::tensor::Tensor::cat([relu76_out1, relu79_out1].into(), 1);
        let conv2d80_out1 = self.conv2d80.forward(concat13_out1);
        let constant12_out1: f32 = 0.1f32;
        let mul12_out1 = conv2d80_out1.mul_scalar(constant12_out1);
        let add12_out1 = mul12_out1.add(relu75_out1);
        let relu80_out1 = burn::tensor::activation::relu(add12_out1);
        let conv2d81_out1 = self.conv2d81.forward(relu80_out1.clone());
        let relu81_out1 = burn::tensor::activation::relu(conv2d81_out1);
        let conv2d82_out1 = self.conv2d82.forward(relu80_out1.clone());
        let relu82_out1 = burn::tensor::activation::relu(conv2d82_out1);
        let conv2d83_out1 = self.conv2d83.forward(relu82_out1);
        let relu83_out1 = burn::tensor::activation::relu(conv2d83_out1);
        let conv2d84_out1 = self.conv2d84.forward(relu83_out1);
        let relu84_out1 = burn::tensor::activation::relu(conv2d84_out1);
        let concat14_out1 = burn::tensor::Tensor::cat([relu81_out1, relu84_out1].into(), 1);
        let conv2d85_out1 = self.conv2d85.forward(concat14_out1);
        let constant13_out1: f32 = 0.1f32;
        let mul13_out1 = conv2d85_out1.mul_scalar(constant13_out1);
        let add13_out1 = mul13_out1.add(relu80_out1);
        let relu85_out1 = burn::tensor::activation::relu(add13_out1);
        let conv2d86_out1 = self.conv2d86.forward(relu85_out1.clone());
        let relu86_out1 = burn::tensor::activation::relu(conv2d86_out1);
        let conv2d87_out1 = self.conv2d87.forward(relu85_out1.clone());
        let relu87_out1 = burn::tensor::activation::relu(conv2d87_out1);
        let conv2d88_out1 = self.conv2d88.forward(relu87_out1);
        let relu88_out1 = burn::tensor::activation::relu(conv2d88_out1);
        let conv2d89_out1 = self.conv2d89.forward(relu88_out1);
        let relu89_out1 = burn::tensor::activation::relu(conv2d89_out1);
        let concat15_out1 = burn::tensor::Tensor::cat([relu86_out1, relu89_out1].into(), 1);
        let conv2d90_out1 = self.conv2d90.forward(concat15_out1);
        let constant14_out1: f32 = 0.1f32;
        let mul14_out1 = conv2d90_out1.mul_scalar(constant14_out1);
        let add14_out1 = mul14_out1.add(relu85_out1);
        let relu90_out1 = burn::tensor::activation::relu(add14_out1);
        let conv2d91_out1 = self.conv2d91.forward(relu90_out1.clone());
        let relu91_out1 = burn::tensor::activation::relu(conv2d91_out1);
        let conv2d92_out1 = self.conv2d92.forward(relu90_out1.clone());
        let relu92_out1 = burn::tensor::activation::relu(conv2d92_out1);
        let conv2d93_out1 = self.conv2d93.forward(relu92_out1);
        let relu93_out1 = burn::tensor::activation::relu(conv2d93_out1);
        let conv2d94_out1 = self.conv2d94.forward(relu93_out1);
        let relu94_out1 = burn::tensor::activation::relu(conv2d94_out1);
        let concat16_out1 = burn::tensor::Tensor::cat([relu91_out1, relu94_out1].into(), 1);
        let conv2d95_out1 = self.conv2d95.forward(concat16_out1);
        let constant15_out1: f32 = 0.1f32;
        let mul15_out1 = conv2d95_out1.mul_scalar(constant15_out1);
        let add15_out1 = mul15_out1.add(relu90_out1);
        let relu95_out1 = burn::tensor::activation::relu(add15_out1);
        relu95_out1
    }
}
