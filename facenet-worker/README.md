# MNIST on Cloudflare Workers
Example "Hello World" using WASM + ML (with `burn`) on [Cloudflare Workers](https://developers.cloudflare.com/workers/).

(Note, you can not take advantage of Cloudflare's GPUs just yet as those are only available from [Workers AI](https://developers.cloudflare.com/workers-ai/) and you can't upload custom models just yet. Still, useful for smaller models you want to run on the edge!)

A previous attempt of mine was trying to compile a binary to WASM and instantiate it in my JS worker. That brings quite a few more headaches when trying to glue the two together. Turns out you can use `worker-rs` instead and write the entire thing in Rust.

Basically, instead of bringing the Rust code to JS with WASM, we can bring the JS Workers bindings to Rust.

# Basics

## Model
There's a demo MNIST classifier (`src/model.rs`) written using `burn`. It's identical to how you would write it for native targets.

## Worker
The worker code lives in `src/lib.rs` and it's very simple. It handles incoming requests by either serving our HTML/JS files (the client) or by relaying the request to the [Durable Object](https://developers.cloudflare.com/durable-objects/) so it can use the classifier model.

## Durable Object
The **DO** is also `src/lib.rs` and it's fairly simple too. The **DO** will fetch the model weights from an [R2](https://developers.cloudflare.com/r2/) bucket, initialize the model and keep it around for further requests until the **DO** is evicted from memory.
Effectively, the model weights are small enough that they could be included in the compiled binary and remove the need to store them in R2 (to avoid binary size/startup limits) but by keeping it like this you can easily scale your model to bigger architectures.

If you're unfamiliar with Durable Objects, have a look at the repo [ystp](https://github.com/deathbyknowledge/ystp) and its blog posts ([part 1](https://deathbyknowledge.com/posts/ystp-pt1/), [part 2](https://deathbyknowledge.com/posts/ystp-pt2/) and [part 3](https://deathbyknowledge.com/posts/ystp-pt3/)).

## HTML/JS client 
There's also a small HTML/JS client to hand draw numbers and display the classifier's ouput probabilities (by calling the worker). Just like the model code, they have been adapted from `burn`'s WASM [demo](https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web). These are in the `public` folder.

## Weights
The weights are the same from the `burn` demo and are located in `model/mnist.bin`.

# Running
To compile and spin up the worker environment just run `npx wrangler dev`. You will need to update the model weights to your local R2, you can do so with `npx wrangler r2 object put models/mnist.bin --local --file model/mnist.bin`. (Were you to deploy this, you'd need to upload to your _real_ R2 bucket, which you can do by removing the `--local` flag)

A deployed version of this can be found in https://mnist-wasm.dbk.wtf/.