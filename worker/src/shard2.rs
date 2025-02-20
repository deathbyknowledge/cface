// Generated from ONNX "pytorch/shard2.onnx" by burn-import
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::record::HalfPrecisionSettings;
use burn::record::Recorder;
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    maxpool2d1: MaxPool2d,
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
    globalaveragepool1: AdaptiveAvgPool2d,
    matmul1: Linear<B>,
    batchnormalization1: BatchNorm<B, 0>,
    phantom: core::marker::PhantomData<B>,
    device: burn::module::Ignored<B::Device>,
}


impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([896, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([256, 384], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([896, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([896, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .init();
        let conv2d8 = Conv2dConfig::new([1792, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([1792, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([192, 192], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([192, 192], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([384, 1792], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([1792, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([1792, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([192, 192], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([192, 192], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([384, 1792], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([1792, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([1792, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([192, 192], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([192, 192], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([384, 1792], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([1792, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([1792, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([192, 192], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([192, 192], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([384, 1792], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([1792, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([1792, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([192, 192], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([192, 192], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([384, 1792], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([1792, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([1792, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([192, 192], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([192, 192], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([384, 1792], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let globalaveragepool1 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let matmul1 = LinearConfig::new(1792, 512).with_bias(false).init(device);
        let batchnormalization1 = BatchNormConfig::new(512)
            .with_epsilon(0.001)
            .with_momentum(0.01)
            .init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            maxpool2d1,
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
            globalaveragepool1,
            matmul1,
            batchnormalization1,
            phantom: core::marker::PhantomData,
            device: burn::module::Ignored(device.clone()),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, input1: Tensor<B, 4>) -> Tensor<B, 2> {
        let conv2d1_out1 = self.conv2d1.forward(input1.clone());
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(input1.clone());
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(input1.clone());
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu6_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(input1);
        let concat1_out1 = burn::tensor::Tensor::cat(
            [relu2_out1, relu4_out1, relu7_out1, maxpool2d1_out1].into(),
            1,
        );
        let conv2d8_out1 = self.conv2d8.forward(concat1_out1.clone());
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = self.conv2d9.forward(concat1_out1.clone());
        let relu9_out1 = burn::tensor::activation::relu(conv2d9_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu9_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu10_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let concat2_out1 = burn::tensor::Tensor::cat([relu8_out1, relu11_out1].into(), 1);
        let conv2d12_out1 = self.conv2d12.forward(concat2_out1);
        let constant1_out1: f32 = 0.2f32;
        let mul1_out1 = conv2d12_out1.mul_scalar(constant1_out1);
        let add1_out1 = mul1_out1.add(concat1_out1);
        let relu12_out1 = burn::tensor::activation::relu(add1_out1);
        let conv2d13_out1 = self.conv2d13.forward(relu12_out1.clone());
        let relu13_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu12_out1.clone());
        let relu14_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu14_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let conv2d16_out1 = self.conv2d16.forward(relu15_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let concat3_out1 = burn::tensor::Tensor::cat([relu13_out1, relu16_out1].into(), 1);
        let conv2d17_out1 = self.conv2d17.forward(concat3_out1);
        let constant2_out1: f32 = 0.2f32;
        let mul2_out1 = conv2d17_out1.mul_scalar(constant2_out1);
        let add2_out1 = mul2_out1.add(relu12_out1);
        let relu17_out1 = burn::tensor::activation::relu(add2_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu17_out1.clone());
        let relu18_out1 = burn::tensor::activation::relu(conv2d18_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu17_out1.clone());
        let relu19_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let conv2d20_out1 = self.conv2d20.forward(relu19_out1);
        let relu20_out1 = burn::tensor::activation::relu(conv2d20_out1);
        let conv2d21_out1 = self.conv2d21.forward(relu20_out1);
        let relu21_out1 = burn::tensor::activation::relu(conv2d21_out1);
        let concat4_out1 = burn::tensor::Tensor::cat([relu18_out1, relu21_out1].into(), 1);
        let conv2d22_out1 = self.conv2d22.forward(concat4_out1);
        let constant3_out1: f32 = 0.2f32;
        let mul3_out1 = conv2d22_out1.mul_scalar(constant3_out1);
        let add3_out1 = mul3_out1.add(relu17_out1);
        let relu22_out1 = burn::tensor::activation::relu(add3_out1);
        let conv2d23_out1 = self.conv2d23.forward(relu22_out1.clone());
        let relu23_out1 = burn::tensor::activation::relu(conv2d23_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu22_out1.clone());
        let relu24_out1 = burn::tensor::activation::relu(conv2d24_out1);
        let conv2d25_out1 = self.conv2d25.forward(relu24_out1);
        let relu25_out1 = burn::tensor::activation::relu(conv2d25_out1);
        let conv2d26_out1 = self.conv2d26.forward(relu25_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv2d26_out1);
        let concat5_out1 = burn::tensor::Tensor::cat([relu23_out1, relu26_out1].into(), 1);
        let conv2d27_out1 = self.conv2d27.forward(concat5_out1);
        let constant4_out1: f32 = 0.2f32;
        let mul4_out1 = conv2d27_out1.mul_scalar(constant4_out1);
        let add4_out1 = mul4_out1.add(relu22_out1);
        let relu27_out1 = burn::tensor::activation::relu(add4_out1);
        let conv2d28_out1 = self.conv2d28.forward(relu27_out1.clone());
        let relu28_out1 = burn::tensor::activation::relu(conv2d28_out1);
        let conv2d29_out1 = self.conv2d29.forward(relu27_out1.clone());
        let relu29_out1 = burn::tensor::activation::relu(conv2d29_out1);
        let conv2d30_out1 = self.conv2d30.forward(relu29_out1);
        let relu30_out1 = burn::tensor::activation::relu(conv2d30_out1);
        let conv2d31_out1 = self.conv2d31.forward(relu30_out1);
        let relu31_out1 = burn::tensor::activation::relu(conv2d31_out1);
        let concat6_out1 = burn::tensor::Tensor::cat([relu28_out1, relu31_out1].into(), 1);
        let conv2d32_out1 = self.conv2d32.forward(concat6_out1);
        let constant5_out1: f32 = 0.2f32;
        let mul5_out1 = conv2d32_out1.mul_scalar(constant5_out1);
        let add5_out1 = mul5_out1.add(relu27_out1);
        let relu32_out1 = burn::tensor::activation::relu(add5_out1);
        let conv2d33_out1 = self.conv2d33.forward(relu32_out1.clone());
        let relu33_out1 = burn::tensor::activation::relu(conv2d33_out1);
        let conv2d34_out1 = self.conv2d34.forward(relu32_out1.clone());
        let relu34_out1 = burn::tensor::activation::relu(conv2d34_out1);
        let conv2d35_out1 = self.conv2d35.forward(relu34_out1);
        let relu35_out1 = burn::tensor::activation::relu(conv2d35_out1);
        let conv2d36_out1 = self.conv2d36.forward(relu35_out1);
        let relu36_out1 = burn::tensor::activation::relu(conv2d36_out1);
        let concat7_out1 = burn::tensor::Tensor::cat([relu33_out1, relu36_out1].into(), 1);
        let conv2d37_out1 = self.conv2d37.forward(concat7_out1);
        let constant6_out1: f32 = 1f32;
        let mul6_out1 = conv2d37_out1.mul_scalar(constant6_out1);
        let add6_out1 = mul6_out1.add(relu32_out1);
        let globalaveragepool1_out1 = self.globalaveragepool1.forward(add6_out1);
        let reshape1_out1 = globalaveragepool1_out1.reshape([1, -1]);
        let matmul1_out1 = self.matmul1.forward(reshape1_out1);
        let batchnormalization1_out1 = self.batchnormalization1.forward(matmul1_out1);
        batchnormalization1_out1
    }
}
