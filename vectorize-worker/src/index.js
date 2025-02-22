const CORS = {
	'Access-Control-Allow-Origin': '*',
	'Access-Control-Allow-Headers': '*',
	'Access-Control-Allow-Methods': 'HEAD, POST, OPTIONS',
};

export default {
	/**
	 * @param {Request} request
	 */
	async scheduled(event, env, ctx) {
		console.log('[CRON] Starting scheduled cleanup...');
		const { objects } = await env.FACES.list();
		const keys = objects.map((obj) => obj.key);
		console.log(`[CRON] found ${keys.length} keys to delete. Deleting...`);
		try {
			await env.FACES.delete(keys);
			await env.VECTORIZE.deleteByIds(keys);
		} catch (e) {
			console.error(`[CRON] Error while trying to delete. Error: ${e}`);
		}
	},

	async fetch(request, env, ctx) {
		switch (request.method) {
			case 'OPTIONS':
				return new Response(null, {
					status: 204, // MDN docs state this should be a 204 code
					headers: {
						...CORS,
					},
				});

			case 'POST': {
				// Handle standard OPTIONS request.
				if (!request.body) return new Response('No body present in the request', { status: 400 });
				let body;
				try {
					body = await request.json();
				} catch (e) {
					return new Response('invalid json', { status: 400 });
				}
				const { name, picture, embeddings } = body;

				const { pathname } = new URL(request.url);
				if (pathname == '/add') {
					if (!name || typeof name != 'string' || !picture || !embeddings) {
						return new Response('missing params', { status: 400 });
					}

					let id = crypto.randomUUID();
					let insert = env.VECTORIZE.upsert([{ id, values: embeddings, metadata: { name } }]);
					let upload = env.FACES.put(id, new Uint8Array(picture));
					await Promise.all([insert, upload]);

					return new Response('ok', { status: 201, headers: CORS }, null, 2);
				} else if (pathname == '/match') {
					if (!embeddings) return new Response('missing params', { status: 400 });
					const { matches } = await env.VECTORIZE.query(embeddings, {
						topK: 1,
						returnMetadata: 'all',
					});

					if (!matches || matches.length == 0) return new Response('no match found', { status: 404 });
					const {
						id,
						score,
						metadata: { name },
					} = matches[0];
					const object = await env.FACES.get(id);
					const picture = Array.from(new Uint8Array(await object.arrayBuffer()));
					return Response.json({ name, picture, score }, { headers: CORS });
				}

				return new Response(null, { status: 404 });
			}
			default:
				return new Response(null, { status: 405 });
		}
	},
};
