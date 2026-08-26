export type Settings = {
  enabled: boolean;
  endpoint: string;
  processorEndpoint: string;
  apiKey: string;
  minWidth: number;
  minHeight: number;
  captureBackgrounds: boolean;
  maxBytes: number;
  skipSvg: boolean;
};

export const DEFAULT_SETTINGS: Settings = {
  enabled: true,
  endpoint: "https://hv.ax",
  processorEndpoint: "",
  apiKey: "",
  minWidth: 64,
  minHeight: 64,
  captureBackgrounds: true,
  maxBytes: 20 * 1024 * 1024,
  skipSvg: true,
};

export type Stats = {
  seen: number;
  posted: number;
  stored: number;
  ignored: number;
  duplicates: number;
  errors: number;
  lastError: string;
  lastStatus: string;
};

export const EMPTY_STATS: Stats = {
  seen: 0,
  posted: 0,
  stored: 0,
  ignored: 0,
  duplicates: 0,
  errors: 0,
  lastError: "",
  lastStatus: "",
};

export type IngestUrlMessage = {
  type: "ingest-url";
  url: string;
  pageUrl: string;
  width: number;
  height: number;
};

export type IngestBytesMessage = {
  type: "ingest-bytes";
  sourceUrl: string;
  pageUrl: string;
  mime: string;
  width: number;
  height: number;
  dataBase64: string;
};

export type GetStatusMessage = { type: "get-status" };
export type ResetStatsMessage = { type: "reset-stats" };

export type ExtMessage = IngestUrlMessage | IngestBytesMessage | GetStatusMessage | ResetStatsMessage;

export type StatusResponse = {
  settings: Settings;
  stats: Stats;
  pending: number;
  serverStats: ServerStats | null;
  serverError: string;
};

export type ServerStats = {
  faces: number;
  images: number;
  embedding_rows: number;
  hnsw: boolean;
};
