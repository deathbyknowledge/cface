// test.mjs
import assert from "node:assert";
import fs from 'node:fs/promises';
import test, { after, before, describe } from "node:test";
import { Miniflare } from "miniflare";

describe("worker", () => {
  /**
   * @type {Miniflare}
   */
  let worker;
  before(async () => {
    // start miniflare
    worker = new Miniflare({
      scriptPath: "./build/worker/shim.mjs",
      modules: true,
      modulesRules: [
        { type: "CompiledWasm", include: ["**/*.wasm"], fallthrough: true },
      ],
      durableObjects: {
        SHARD1: { className: "FaceNetShard1" },
        SHARD2: { className: "FaceNetShard2" }
      },
      r2Buckets: {
        MODELS: "models"
      },
    });
    await worker.ready;

    // put model weights in miniflare's r2
    let models = await worker.getR2Bucket("MODELS");
    const shard1 = await fs.readFile("model/shard1.bin");
    const shard2 = await fs.readFile("model/shard2.bin");
    try {
      await models.put("shard1.bin", new Uint8Array(shard1));
      await models.put("shard2.bin", new Uint8Array(shard2));
    } catch (e) { 
     console.error(e);
      process.exit(1);
    }
    await worker.ready;
  });

  test("404s", async () => {
    const res = await worker.dispatchFetch("http://localhost/");
    assert(res.status == 404);
  });

  test("405s", async () => {
    const res = await worker.dispatchFetch("http://localhost/compute");
    assert(res.status == 405);
  });

  test("computes face descriptors", async () => {
    const data = await fs.readFile("tests/cropped-face.jpeg");
    let res = await worker.dispatchFetch("http://localhost/compute", {
      body: data,
      method: "POST"
    });
    assert(res.status == 200);
    const embeddings = await res.json();
    assert(embeddings.length == 512);
    res = await worker.dispatchFetch("http://localhost/compute", {
      body: data,
      method: "POST"
    });
    assert(res.status == 200);
    const embeddings2 = await res.json();
    assert(embeddings[0] == embeddings2[0]);
    assert(embeddings[511] == embeddings2[511]);
  });

  after(async () => {
    await worker.dispose();
  });
});

