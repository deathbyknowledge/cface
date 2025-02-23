pub mod durable_object;
pub mod shard1;
pub mod shard2;

use worker::*;

#[event(fetch)]
async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    console_error_panic_hook::set_once();
    let cors = Cors::default()
        .with_max_age(86400)
        .with_origins(vec!["*"])
        .with_allowed_headers(vec!["*"])
        .with_methods(vec![Method::Post, Method::Options]);

    // Durable Objects get geographically pinned, so we'll instantiate
    // them by the source continent of the request.
    if req.path() == "/compute" {
        match req.method() {
            // Read request's continent
            Method::Post => {
                let continent = req
                    .cf()
                    .expect("Failed to read CF request info")
                    .continent()
                    .expect("Failed to read CF Continent");

                let model_runner = env
                    .durable_object("SHARD1")?
                    .id_from_name(&continent)?
                    .get_stub()?;
                match model_runner.fetch_with_request(req).await {
                    Ok(res) => return res.with_cors(&cors),
                    Err(e) => {
                        console_error!("[Worker] Received error while computing: {e}");
                        return Response::error("error ocurred when computing image", 500);
                    }
                }
            }
            Method::Options => {
                return Response::builder()
                    .with_status(204)
                    .empty()
                    .with_cors(&cors);
            }

            _ => return Response::error("", 405),
        }
    }
    Response::error("", 404)
}
