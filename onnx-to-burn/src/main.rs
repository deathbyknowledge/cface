use burn_import::onnx::ModelGen;

fn main() {
    // Make sure the you've run `cd pytorch && python facenet.py`
    ModelGen::new()
        .input("pytorch/shard1.onnx")
        .half_precision(true)
        .record_type(burn_import::onnx::RecordType::Bincode)
        .out_dir("out/")
        .run_from_cli();

    ModelGen::new()
        .input("pytorch/shard2.onnx")
        .half_precision(true)
        .record_type(burn_import::onnx::RecordType::Bincode)
        .out_dir("out/")
        .run_from_cli();
}
