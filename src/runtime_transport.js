// Normalize only the published large-model transfer; model.js still verifies its SHA256.
// Some static gateways serve a precompressed payload without Content-Encoding.
export function createRuntimeFetch(nativeFetch, assetURL) {
  const target = new URL(assetURL);
  return async function runtimeFetch(input, init) {
    const url = new URL(input instanceof Request ? input.url : input, target);
    const response = await nativeFetch(input, init);
    if (url.origin !== target.origin || url.pathname !== target.pathname || !response.ok)
      return response;
    const bytes = await response.arrayBuffer();
    const signature = new Uint8Array(bytes, 0, Math.min(2, bytes.byteLength));
    let decoded = bytes;
    if (signature[0] === 0x1f && signature[1] === 0x8b) {
      if (typeof DecompressionStream !== "function")
        throw Error("This browser cannot decode the face model transfer. Update the browser and retry.");
      try {
        decoded = await new Response(
          new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip")),
        ).arrayBuffer();
      } catch {
        throw Error("The face model transfer is incomplete or damaged. Reload to retry.");
      }
    }
    const headers = new Headers(response.headers);
    headers.delete("content-encoding");
    headers.delete("content-length");
    headers.delete("etag");
    headers.set("content-type", "application/octet-stream");
    return new Response(decoded, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  };
}
