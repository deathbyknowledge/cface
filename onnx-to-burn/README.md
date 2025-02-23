# Export ONNX model to `burn` code
First, run the `facenet.py` file in `./pytorch`, it will export the (sharded) FaceNet model from `.pt` format to onnx. You'll then have 2 `.onnx` files.

You can then run `cargo run` to generate both the `burn` definitions of the model + the weights file in `burn`'s binary format. They'll be output in `./out`.
