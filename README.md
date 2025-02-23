# CFace
Searchable facial database built entirely on the Cloudflare stack. No server required.

[https://cface.defo.works/](https://cface.defo.works/)


## What's here?

The idea is to have everything fit in Cloudflare's edge compute. So let's start by listing the requirements needed to build it. 

- **A model to compute face descriptors on face images** - This one is quite big (~90MB), so we'll run it on the edge.
- **A model to detect faces on an image** - This one can run on the browser seamlessly, since it's quite small.
- **A vector database where we can store and query the vectors by euclidean distance**. - This will be our search engine.
- **A client to use all the above!**. - HTML to run on the browsre is enough.


### The facial recognition model
>  Our method is based on learning a Euclidean embedding per image using a deep convolutional network. The network is trained such that the squared L2 distances in the embedding space directly correspond to face similarity.

From the [FaceNet paper](https://arxiv.org/abs/1503.03832).

The pre-trained weights (from [facenet-pytorch](https://github.com/timesler/facenet-pytorch/)) are around 90MB. If the entire ported model were intialized in the Worker, it would go over the 128MB memory limit, so we split the model into 2 shards.

Cloudflare Workers have great WASM support and you can write them all in Rust without worrying about WASM bindings too muchby using `workers-rs`, so here we port the model from `torch` (python) to `burn` (Rust).

`onnx-to-burn/pytorch/model.py` has the split model definition. `extract.py` loads the model weights and then exports the shards in `.onnx` format.

Then `onnx-to-burn/main.rs` reads the `.onnx` files generated and outputs the 2 model files using `shard1.rs` and `shard2.rs`. It also outputs the model weights in f16 precision in `burn`'s binary format, `shard1.bin` and `shard2.bin`. (These last 2 files we upload to R2, so we can fetch them on the edge).

The model files can now be used in the Worker/Durable Object code which will run inference, located in `facenet-worker/`.

### The facial deteciton model
Not much to say here, the client uses [face-api.js](https://justadudewhohacks.github.io/face-api.js/docs/index.html) to detect faces and make them 160x160 (the input size to the recognition model) to send them to the Worker running the recognition.


### Vector database
Cloudflare's Vectorize is a perfect fit. We just need to upsert and query the index, which is in `vectorize-worker/src/index.js`.

This same worker also has a cron trigger that nukes the vector index and the R2 bucket (with face images) every day at 12:00.

### HTML Client
Putting it all together, just an HTML file in `site/index.html` where you can take a picture of yourself and upload it or match against the database. It is hosted as a simple static site.

We host the face detection model weights (small) in `site/models`.

