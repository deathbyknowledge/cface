pub mod model;
pub mod shard1;
pub mod shard2;
pub mod durable_object;

use worker::*;

#[event(fetch)]
async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    console_error_panic_hook::set_once();

    // Durable Objects get geographically pinned, so we'll instantiate
    // them by the source continent of the request.
    if req.method() == Method::Post && req.path().contains("/compute") {
        // Read request's continent
        let continent = req
            .cf()
            .expect("Failed to read CF request info")
            .continent()
            .expect("Failed to read CF Continent");

        let model_runner = env
            .durable_object("SHARD1")?
            .id_from_name(&continent)?
            .get_stub()?;
        return model_runner.fetch_with_request(req).await;
    } else {
        return env.assets("ASSETS")?.fetch_request(req).await;
    }
}
