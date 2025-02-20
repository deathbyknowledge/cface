use burn_import::onnx::ModelGen;

fn main() {
        // If not embedded-model, then model is loaded from the file system (default).
        ModelGen::new()
            .input("src/model/resnet.onnx")
            .half_precision(true)
            .record_type(burn_import::onnx::RecordType::Bincode)
            .out_dir("src/model/")
            .run_from_cli();
}
