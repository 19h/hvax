export type ApiPath = "/v1/ingest" | "/v1/stats" | "/health";

export function resolveApiUrl(configuredUrl: string, apiPath: ApiPath): string {
  const url = new URL(configuredUrl);
  const pathname = url.pathname.replace(/\/+$/, "");
  const apiMarker = pathname.lastIndexOf("/v1/");
  const basePath = apiMarker >= 0 ? pathname.slice(0, apiMarker) : pathname;

  url.pathname = `${basePath}${apiPath}`;
  url.hash = "";
  return url.href;
}
