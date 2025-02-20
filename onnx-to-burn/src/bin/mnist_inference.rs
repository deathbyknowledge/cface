use std::env::args;

use burn::{
    backend::ndarray::NdArray,
    tensor::Tensor,
};
use image::imageops::FilterType;

use onnx_to_burn::resnet::Model;

const IMAGE_PATH: &str = "./cropped.jpg"; // <- Change this to test a different image

fn main() {
    // Get image index argument (first) from command line

    let image_path = if let Some(image_path) = args().nth(1) {
        println!("Image selected: {}", image_path);
        image_path
    } else {
        println!("No image index provided; Using default image index: {IMAGE_PATH}");
        IMAGE_PATH.to_string()
    };

    let (batch, channels, height, width) = (1, 3, 160, 160);

    // Load the image as an RgbImage
    let img = image::open(image_path)
        .expect("Failed to open image")
        .to_rgb8();

    // Optionally resize to 160x160
    let resized_img = image::imageops::resize(&img, width, height, FilterType::CatmullRom);

    // Convert to Vec<f32>, normalized to [0.0, 1.0]
    // The output layout will be [R0, G0, B0, R1, G1, B1, ..., Rn, Gn, Bn]
    let mut rgb_data = Vec::with_capacity((width * height * channels) as usize);
    for pixel in resized_img.pixels() {
        // Each pixel: [R, G, B]
        rgb_data.push(pixel[0] as f32 / 255.0);
        rgb_data.push(pixel[1] as f32 / 255.0);
        rgb_data.push(pixel[2] as f32 / 255.0);
    }

    let mut nchw_data = vec![0.0; (width * height * channels) as usize];
    // For each pixel index i, place it in the correct channel offset.
    for row in 0..height {
        for col in 0..width {
            let pixel_i = (row * width + col) as usize;
            // R channel at offset 0
            nchw_data[0 * (width as usize * height as usize) + pixel_i] = rgb_data[pixel_i * 3 + 0];
            // G channel at offset 1
            nchw_data[1 * (width as usize * height as usize) + pixel_i] = rgb_data[pixel_i * 3 + 1];
            // B channel at offset 2
            nchw_data[2 * (width as usize * height as usize) + pixel_i] = rgb_data[pixel_i * 3 + 2];
        }
    }
    type Backend = NdArray<f32>;

    // Get a default device for the backend
    let device = <Backend as burn::tensor::backend::Backend>::Device::default();

    let input =
        Tensor::<Backend, 1>::from_floats(nchw_data.as_slice(), &device).reshape([1, 3, 160, 160]);

    // Create a new model and load the state
    let model: Model<Backend> = Model::new(&Default::default());

    // Run the model on the input
    let output = model.forward(input);
    let norm = output.clone().powf_scalar(2.0).sum_dim(1).sqrt();
    let normalized = output.div(norm);

    let embdgs: Vec<f32> = normalized.into_data().to_vec().unwrap();

    println!("Success!");
    println!("L2-normalized embedding from Burn = {:#?}", embdgs);
    // println!("Got: {:#?}", embdgs);
}
