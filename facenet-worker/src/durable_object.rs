pub type Backend = burn::backend::ndarray::NdArray<f32>;

pub mod shard1 {
    use super::Backend;
    use crate::model::shard1::Model as Shard1;
    use burn::tensor::Tensor;
    use image::imageops::FilterType;
    use image::ImageReader;
    use js_sys;
    use wasm_bindgen::prelude::*;
    use worker::*;

    #[durable_object]
    pub struct FaceNetShard1 {
        model: Option<Shard1<Backend>>,
        state: State,
        env: Env,
    }

    #[durable_object]
    impl DurableObject for FaceNetShard1 {
        fn new(state: State, env: Env) -> Self {
            Self {
                model: None,
                state,
                env,
            }
        }

        async fn fetch(&mut self, mut req: Request) -> Result<Response> {
            if self.model.is_none() {
                self.load_model().await?;
            }

            // Expects it to be a 160x160 JPEG
            let bytes: Vec<u8> = req.bytes().await?;

            // Bytes of the tensor where this shard ends.
            let input = self.image_bytes_to_vec(bytes)?;
            let result = self.compute(&input);

            let continent = req
                .cf()
                .expect("Failed to read CF request info")
                .continent()
                .expect("Failed to read CF Continent");

            // Currently, to add a body to a Request, we need to convert
            // to a JsValue first.
            let uint8array = js_sys::Uint8Array::from(result.as_slice());
            let js_value: JsValue = uint8array.into();

            let shard_req = Request::new_with_init(
                &req.url()?.to_string(),
                RequestInit::new()
                    .with_body(Some(js_value))
                    .with_method(Method::Post),
            )?;

            let shard2 = self
                .env
                .durable_object("SHARD2")?
                .id_from_name(&continent)?
                .get_stub()?;

            return shard2.fetch_with_request(shard_req).await;
        }
    }

    impl FaceNetShard1 {
        /// Classify the input image [f32; 28*28] and return the array of probabilities.
        fn compute(&mut self, input: &[f32]) -> Vec<u8> {
            let device = Default::default();
            let input = Tensor::<Backend, 1>::from_floats(input, &device).reshape([1, 3, 160, 160]);

            let bytes = {
                let output = self.model.as_ref().unwrap().forward(input);
                let tensor_data = output.into_data();
                tensor_data.as_bytes().to_vec()
            };

            bytes
        }

        /// Fetch model weights from R2 and load the model into the DO
        async fn load_model(&mut self) -> Result<()> {
            use burn::{
                module::Module,
                record::{BinBytesRecorder, HalfPrecisionSettings, Recorder},
            };

            let id = self.state.id().to_string();
            console_log!(
                "[Shard1] No model found in memory. Loading it from R2.\nInstance id: {id}"
            );
            // 1. Fetch bytes
            let bytes = {
                let bucket = self.env.bucket("MODELS")?;
                let obj = bucket
                    .get("shard1.bin")
                    .execute()
                    .await?
                    .expect("Model not found");
                obj.body().expect("No body").bytes().await?
            };

            let model: Shard1<Backend> = Shard1::new(&Default::default());
            let record = {
                let record = BinBytesRecorder::<HalfPrecisionSettings>::default()
                    .load(bytes, &Default::default())
                    .expect("Failed to decode state");
                record
            };

            self.model = Some(model.load_record(record));
            console_log!("[Shard1] Successfully loaded model weights from R2.\nInstance id: {id}");
            Ok(())
        }

        fn image_bytes_to_vec(&self, bytes: Vec<u8>) -> Result<Vec<f32>> {
            let img = ImageReader::new(std::io::Cursor::new(&bytes))
                .with_guessed_format()?
                .decode()
                .map_err(|err| err.to_string())?
                .resize_to_fill(160, 160, FilterType::Nearest)
                .to_rgb8();

            let (_batch, channels, height, width) = (1, 3, 160, 160);

            // Convert to Vec<f32>, normalized to [0.0, 1.0]
            // The output layout will be [R0, G0, B0, R1, G1, B1, ..., Rn, Gn, Bn]
            let mut rgb_data = Vec::with_capacity((width * height * channels) as usize);
            for pixel in img.pixels() {
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
                    nchw_data[0 * (width as usize * height as usize) + pixel_i] =
                        rgb_data[pixel_i * 3 + 0];
                    // G channel at offset 1
                    nchw_data[1 * (width as usize * height as usize) + pixel_i] =
                        rgb_data[pixel_i * 3 + 1];
                    // B channel at offset 2
                    nchw_data[2 * (width as usize * height as usize) + pixel_i] =
                        rgb_data[pixel_i * 3 + 2];
                }
            }
            Ok(nchw_data)
        }
    }
}

pub mod shard2 {
    use super::Backend;
    use crate::model::shard2::Model as Shard2;
    use burn::tensor::{DType, Tensor, TensorData};
    use worker::*;

    #[durable_object]
    pub struct FaceNetShard2 {
        model: Option<Shard2<Backend>>,
        state: State,
        env: Env,
    }

    #[durable_object]
    impl DurableObject for FaceNetShard2 {
        fn new(state: State, env: Env) -> Self {
            Self {
                model: None,
                state,
                env,
            }
        }

        async fn fetch(&mut self, mut req: Request) -> Result<Response> {
            if self.model.is_none() {
                self.load_model().await?;
            }

            // Expects it to be the output of Shard1's last layer
            let bytes: Vec<u8> = req.bytes().await?;

            // [f32; 512] face embeddings
            let result = self.compute(bytes);

            Response::from_json(&result)
        }
    }

    impl FaceNetShard2 {
        fn compute(&mut self, input: Vec<u8>) -> Vec<f32> {
            let device = Default::default();
            let data = TensorData::from_bytes(input, [1, 896, 8, 8], DType::F32);
            let input = Tensor::<Backend, 4>::from_data(data, &device);

            let output = self.model.as_ref().unwrap().forward(input);

            let norm = output.clone().powf_scalar(2.0).sum_dim(1).sqrt();
            let normalized = output.div(norm);

            let embdgs: Vec<f32> = normalized.into_data().to_vec().unwrap();

            embdgs
        }

        /// Fetch model weights from R2 and load the model into the DO
        async fn load_model(&mut self) -> Result<()> {
            use burn::{
                module::Module,
                record::{BinBytesRecorder, HalfPrecisionSettings, Recorder},
            };

            let id = self.state.id().to_string();
            console_log!(
                "[Shard1] No model found in memory. Loading it from R2.\nInstance id: {id}"
            );

            let bytes = {
                let bucket = self.env.bucket("MODELS")?;
                let obj = bucket
                    .get("shard2.bin")
                    .execute()
                    .await?
                    .expect("Model not found");
                obj.body().expect("No body").bytes().await?
            };

            let model: Shard2<Backend> = Shard2::new(&Default::default());
            let record = {
                let record = BinBytesRecorder::<HalfPrecisionSettings>::default()
                    .load(bytes, &Default::default())
                    .expect("Failed to decode state");
                record
            };

            self.model = Some(model.load_record(record));
            console_log!("[Shard2] Successfully loaded model weights from R2.\nInstance id: {id}");
            Ok(())
        }
    }
}
