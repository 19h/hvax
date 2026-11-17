import type { Settings } from "./types";

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

function hostnameOf(raw: string): string {
  try {
    return new URL(raw).hostname.toLowerCase().replace(/\.$/, "");
  } catch {
    return "";
  }
}

/**
 * Hosts owned by the configured hvax backends. An image served from one of
 * these hosts is already in (or behind) the gallery, so sending it back to
 * `/v1/ingest` would only produce a duplicate. Subdomains are included so a
 * gallery that serves assets from e.g. `cdn.hv.ax` is covered too.
 */
export function backendHosts(settings: Pick<Settings, "endpoint" | "processorEndpoint">): string[] {
  const hosts = new Set<string>();
  for (const configured of [settings.endpoint, settings.processorEndpoint]) {
    const host = hostnameOf(configured);
    if (host) hosts.add(host);
  }
  return [...hosts];
}

export function isBackendImage(url: string, settings: Pick<Settings, "endpoint" | "processorEndpoint">): boolean {
  const host = hostnameOf(url);
  if (!host) return false;
  return backendHosts(settings).some((backend) => host === backend || host.endsWith(`.${backend}`));
}
