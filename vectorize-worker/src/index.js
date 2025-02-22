const corsHeaders = {
	'Access-Control-Allow-Origin': '*',
	'Access-Control-Allow-Methods': 'HEAD, POST, OPTIONS',
	'Access-Control-Max-Age': '86400', // optional
	'Access-Control-Allow-Headers': '*',
};

export default {
	/**
	 * @param {Request} request
	 */
	async scheduled(event, env, ctx) {
		const { objects } = await env.FACES.list();
		const keys = objects.map((obj) => obj.key);
		await env.FACES.delete(keys);
		await env.VECTORIZE.deleteByIds(keys);
	},

	async fetch(request, env, ctx) {
		if (request.method == 'OPTIONS') {
			// Handle standard OPTIONS request.
			return new Response(null, {
				status: 204, // MDN docs state this should be a 204 code
				headers: {
					...corsHeaders,
				},
			});
		}
		const { name, picture, embeddings } = await request.json();

		if (request.url.includes('/add')) {
			if (!name || typeof name != 'string' || !picture || !embeddings) {
				return new Response('missing params', { status: 400 });
			}

			let id = crypto.randomUUID();
			let insert = env.VECTORIZE.upsert([{ id, values: embeddings, metadata: { name } }]);
			let upload = env.FACES.put(id, new Uint8Array(picture));
			await Promise.all([insert, upload]);

			return new Response('ok', { status: 201, headers: corsHeaders }, null, 2);
		} else if (request.url.includes('/match')) {
			if (!embeddings) return new Response('missing params', { status: 400 });
			const { matches } = await env.VECTORIZE.query(embeddings, {
				topK: 1,
				returnMetadata: 'all',
			});

			if (!matches || matches.length == 0) return new Response('no match found', { status: 404 });
			const {
				id,
				metadata: { name },
			} = matches[0];
			const object = await env.FACES.get(id);
			const picture = Array.from(new Uint8Array(await object.arrayBuffer()));
			return Response.json({ name, picture }, { headers: corsHeaders });
		}
		return new Response('not found', { status: 404 });
	},
};
