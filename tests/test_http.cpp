#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <random>
#include <string_view>
#include <thread>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "httplib.h"
#include "hvax/align/umeyama.hpp"
#include "hvax/config.hpp"
#include "hvax/embed/arcface.hpp"
#include "hvax/engine.hpp"
#include "hvax/http/jobs.hpp"
#include "hvax/http/landing_html.hpp"
#include "hvax/http/server.hpp"
#include "hvax/pipeline.hpp"
#include "hvax/util/hex.hpp"

TEST(Landing, HtmlDocument) {
  const std::string_view html(kLandingHtml);
  EXPECT_NE(html.find("<!DOCTYPE html>"), std::string_view::npos);
  EXPECT_NE(html.find("hvax"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest/pdf"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest/check"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest/processed"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/query/image"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/identities"), std::string_view::npos);
  EXPECT_NE(html.find("/crop"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/query/template"), std::string_view::npos);
  EXPECT_NE(html.find("value=\"32\""), std::string_view::npos);
  EXPECT_NE(html.find("type=\"file\" accept=\"image/*,application/pdf,.pdf\" multiple"),
            std::string_view::npos);
  EXPECT_NE(html.find("sendIngestBatch"), std::string_view::npos);
  EXPECT_NE(html.find("/v1/ingest/pdf?detect_only=1&include_embedding=1"), std::string_view::npos);
  EXPECT_NE(html.find("hasFaceThumbnails"), std::string_view::npos);
  EXPECT_NE(html.find("has-thumb"), std::string_view::npos);
  EXPECT_NE(html.find("not my person"), std::string_view::npos);
  EXPECT_NE(html.find("zoomable"), std::string_view::npos);
  EXPECT_NE(html.find("IntersectionObserver"), std::string_view::npos);
  EXPECT_NE(html.find("MAX_IMAGE_REQUESTS = 4"), std::string_view::npos);
  EXPECT_NE(html.find("function onClipboardPaste"), std::string_view::npos);
  EXPECT_NE(html.find("function collectClipboardFiles"), std::string_view::npos);
  EXPECT_NE(html.find("function clipboardImageUrls"), std::string_view::npos);
  EXPECT_NE(html.find("addEventListener(\"paste\", onClipboardPaste)"), std::string_view::npos);
  EXPECT_NE(html.find("paste or drop images or PDFs"), std::string_view::npos);
}

TEST(Config, DefaultSearchSizeIsThirtyTwo) { EXPECT_EQ(hvax::Config{}.default_k, 32); }

TEST(Http, ListenBacklogHandlesThumbnailBursts) { EXPECT_GE(CPPHTTPLIB_LISTEN_BACKLOG, 128); }

TEST(PipelineHelpers, SniffMime) {
  const uint8_t jpeg[] = {0xff, 0xd8, 0xff, 0x00};
  EXPECT_EQ(hvax::sniff_mime(jpeg), hvax::Mime::jpeg);
  const uint8_t png[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
  EXPECT_EQ(hvax::sniff_mime(png), hvax::Mime::png);
}

TEST(Config, DefaultPixelLimitAcceptsHighResolutionPhotos) {
  EXPECT_GE(hvax::Config{}.max_pixels, int64_t{8480} * 5664);
}

TEST(HttpHelpers, AttachJobsOverwritesRemoteField) {
  const auto out = hvax::attach_jobs(nlohmann::json{{"faces", 1}, {"jobs", 8}}, 4);
  EXPECT_EQ(out.at("jobs").get<int>(), 4);
  EXPECT_EQ(out.at("faces").get<int>(), 1);
}

TEST(HttpHelpers, AttachJobsIgnoresNonObject) {
  const auto out = hvax::attach_jobs(nlohmann::json::array({1, 2}), 4);
  EXPECT_TRUE(out.is_array());
  EXPECT_EQ(out.size(), 2);
}

TEST(HttpHelpers, HashHexRoundTrip) {
  constexpr uint64_t expected = 0xfedcba9876543210ULL;
  const auto encoded = hvax::to_hex64(expected);
  EXPECT_EQ(encoded, "fedcba9876543210");
  uint64_t decoded = 0;
  EXPECT_TRUE(hvax::hex64_from_string(encoded, decoded));
  EXPECT_EQ(decoded, expected);
  EXPECT_FALSE(hvax::hex64_from_string("not-a-valid-hash", decoded));
}

namespace {

std::filesystem::path http_tmpdir() {
  static std::atomic<int> seq{0};
  auto p = std::filesystem::temp_directory_path() /
           ("hvax-http-test-" + std::to_string(::getpid()) + "-" + std::to_string(seq.fetch_add(1)));
  std::filesystem::create_directories(p);
  return p;
}

hvax::Embedding person_embedding(int person, int sample) {
  std::mt19937 base(1000 + person);
  std::normal_distribution<float> nd(0.f, 1.f);
  hvax::Embedding e{};
  for (auto& x : e) x = nd(base);
  std::mt19937 rng(static_cast<unsigned>(person * 100003 + sample));
  for (auto& x : e) x += 0.35f * nd(rng);
  hvax::l2_normalize(e.data());
  return e;
}

hvax::DetectedFace face_at(const hvax::Embedding& e, float x, float y, float size) {
  hvax::DetectedFace f;
  f.box = {x, y, x + size, y + size};
  f.det_score = 0.9f;
  f.kps = hvax::arcface_dst();
  f.embedding = e;
  return f;
}

cv::Mat photo(int seed) {
  cv::Mat img(640, 640, CV_8UC3);
  for (int y = 0; y < 640; ++y)
    for (int x = 0; x < 640; ++x)
      img.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uint8_t>(x + seed), static_cast<uint8_t>(y + seed * 2), 80);
  cv::rectangle(img, {10, 10, 200, 200}, {30, 90, 220}, -1);
  return img;
}

void ingest(hvax::Engine& engine, int seed, const std::vector<hvax::DetectedFace>& faces) {
  cv::Mat img = photo(seed);
  std::vector<uint8_t> bytes;
  cv::imencode(".jpg", img, bytes, {cv::IMWRITE_JPEG_QUALITY, 90});
  auto r = engine.ingest_processed(bytes, img, faces);
  ASSERT_EQ(r.status, hvax::IngestStatus::stored);
}

struct LiveServer {
  hvax::Engine& engine;
  httplib::Server svr;
  std::thread thread;
  int port = 0;
  explicit LiveServer(hvax::Engine& e) : engine(e) {
    hvax::register_routes(engine, svr);
    port = svr.bind_to_any_port("127.0.0.1");
    thread = std::thread([this] { svr.listen_after_bind(); });
    svr.wait_until_ready();
  }
  ~LiveServer() {
    svr.stop();
    thread.join();
  }
  httplib::Client client() { return httplib::Client("127.0.0.1", port); }
};

std::string raw_embedding(const hvax::Embedding& e) {
  return std::string(reinterpret_cast<const char*>(e.data()), sizeof(float) * hvax::kDim);
}

}  // namespace

TEST(HttpApi, IdentityRoutesEndToEnd) {
  auto dir = http_tmpdir();
  hvax::Config cfg;
  cfg.data_dir = dir.string();
  cfg.api_key = "secret";
  cfg.identity_browse = true;
  cfg.dedup = hvax::DedupMode::sha256;  // fixture photos are near-identical; this test is about identities
  hvax::Engine engine(cfg);
  for (int i = 0; i < 3; ++i) {
    ingest(engine, 10 + i, {face_at(person_embedding(1, i), 10, 10, 200)});
    ingest(engine, 20 + i, {face_at(person_embedding(2, i), 10, 10, 200)});
  }
  ingest(engine, 30, {face_at(person_embedding(1, 9), 10, 10, 200), face_at(person_embedding(2, 9), 300, 300, 200),
                      face_at(person_embedding(3, 0), 500, 500, 12)});

  LiveServer live(engine);
  auto c = live.client();
  const httplib::Headers key = {{"X-API-Key", "secret"}};

  // auth is enforced on the new routes
  {
    auto r = c.Get("/v1/identities");
    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 401);
  }
  // stats carry the identity/quality counters
  {
    auto r = c.Get("/v1/stats", key);
    ASSERT_TRUE(r);
    ASSERT_EQ(r->status, 200);
    auto j = nlohmann::json::parse(r->body);
    EXPECT_EQ(j["faces"], 9);
    EXPECT_EQ(j["indexed_faces"], 8);
    EXPECT_EQ(j["low_quality_faces"], 1);
    EXPECT_EQ(j["identities"], 0);
    EXPECT_EQ(j["unassigned_faces"], 8);
    EXPECT_TRUE(j.contains("i8_kernel"));
  }
  // trigger clustering
  int64_t id_a = -1, id_b = -1;
  {
    auto r = c.Post("/v1/identities/cluster", key, "", "application/json");
    ASSERT_TRUE(r);
    ASSERT_EQ(r->status, 200) << r->body;
    auto j = nlohmann::json::parse(r->body);
    EXPECT_EQ(j["identities"], 2);
    EXPECT_EQ(j["faces_assigned"], 8);
    auto f0 = nlohmann::json::parse(c.Get("/v1/faces/0", key)->body);
    auto f1 = nlohmann::json::parse(c.Get("/v1/faces/1", key)->body);
    id_a = f0["identity_id"].get<int64_t>();
    id_b = f1["identity_id"].get<int64_t>();
    EXPECT_NE(id_a, id_b);
    EXPECT_GT(f0["quality"].get<float>(), 0.8f);
    EXPECT_FALSE(f0["low_quality"].get<bool>());
    auto f8 = nlohmann::json::parse(c.Get("/v1/faces/8", key)->body);
    EXPECT_TRUE(f8["low_quality"].get<bool>());
    EXPECT_TRUE(f8["identity_id"].is_null());
  }
  // listing, detail, faces, cooccurring, timeline
  {
    auto j = nlohmann::json::parse(c.Get("/v1/identities?sort=size&limit=10", key)->body);
    EXPECT_EQ(j["total"], 2);
    ASSERT_EQ(j["identities"].size(), 2u);
    EXPECT_EQ(j["identities"][0]["size"], 4);
    auto d = nlohmann::json::parse(c.Get("/v1/identities/" + std::to_string(id_a), key)->body);
    EXPECT_EQ(d["size"], 4);
    EXPECT_EQ(d["n_images"], 4);
    ASSERT_EQ(d["cooccurring"].size(), 1u);
    EXPECT_EQ(d["cooccurring"][0]["identity_id"], id_b);
    EXPECT_EQ(d["cooccurring"][0]["shared_images"], 1);
    auto f = nlohmann::json::parse(c.Get("/v1/identities/" + std::to_string(id_a) + "/faces?limit=2", key)->body);
    ASSERT_EQ(f["faces"].size(), 2u);
    EXPECT_GE(f["faces"][0]["score"].get<float>(), f["faces"][1]["score"].get<float>());
    auto co = nlohmann::json::parse(c.Get("/v1/identities/" + std::to_string(id_a) + "/cooccurring", key)->body);
    ASSERT_EQ(co["cooccurring"].size(), 1u);
    EXPECT_EQ(co["cooccurring"][0]["size"], 4);
    auto tl = nlohmann::json::parse(c.Get("/v1/identities/" + std::to_string(id_a) + "/timeline?bucket=day", key)->body);
    EXPECT_EQ(tl["bucket_ms"], 86400000);
    ASSERT_GE(tl["buckets"].size(), 1u);
    auto miss = c.Get("/v1/identities/999", key);
    EXPECT_EQ(miss->status, 404);
  }
  // face search: grouped, identity mode, range, low-quality inclusion
  {
    httplib::Headers h = key;
    h.emplace("X-K", "5");
    h.emplace("X-Min-Score", "-1");
    h.emplace("X-Group-By", "identity");
    auto r = c.Post("/v1/query/embedding", h, raw_embedding(person_embedding(1, 77)), "application/octet-stream");
    ASSERT_EQ(r->status, 200);
    auto j = nlohmann::json::parse(r->body);
    ASSERT_GE(j["hits"].size(), 2u);
    EXPECT_EQ(j["hits"][0]["identity_id"], id_a);
    EXPECT_EQ(j["hits"][0]["collapsed"], 3);

    httplib::Headers hi = key;
    hi.emplace("X-K", "2");
    hi.emplace("X-Min-Score", "-1");
    hi.emplace("X-Mode", "identity");
    r = c.Post("/v1/query/embedding", hi, raw_embedding(person_embedding(1, 78)), "application/octet-stream");
    ASSERT_EQ(r->status, 200);
    j = nlohmann::json::parse(r->body);
    ASSERT_EQ(j["identities"].size(), 2u);
    EXPECT_EQ(j["identities"][0]["identity_id"], id_a);
    EXPECT_EQ(j["identities"][0]["best_face"]["identity_id"], id_a);

    httplib::Headers hr = key;
    hr.emplace("X-K", "1");
    hr.emplace("X-Min-Score", "0.5");
    hr.emplace("X-Mode", "range");
    r = c.Post("/v1/query/embedding", hr, raw_embedding(person_embedding(1, 79)), "application/octet-stream");
    j = nlohmann::json::parse(r->body);
    EXPECT_EQ(j["hits"].size(), 4u);

    httplib::Headers hq = key;
    hq.emplace("X-Min-Score", "0.5");
    r = c.Post("/v1/query/embedding", hq, raw_embedding(person_embedding(3, 0)), "application/octet-stream");
    EXPECT_EQ(nlohmann::json::parse(r->body)["hits"].size(), 0u);
    hq.emplace("X-Include-Low-Quality", "1");
    r = c.Post("/v1/query/embedding", hq, raw_embedding(person_embedding(3, 0)), "application/octet-stream");
    j = nlohmann::json::parse(r->body);
    ASSERT_EQ(j["hits"].size(), 1u);
    EXPECT_TRUE(j["hits"][0]["low_quality"].get<bool>());
  }
  // crop endpoint returns a decodable square JPEG
  {
    auto r = c.Get("/v1/faces/0/crop?size=64", key);
    ASSERT_TRUE(r);
    ASSERT_EQ(r->status, 200) << r->body;
    EXPECT_EQ(r->get_header_value("Content-Type"), "image/jpeg");
    std::vector<uint8_t> bytes(r->body.begin(), r->body.end());
    cv::Mat img = cv::imdecode(bytes, cv::IMREAD_COLOR);
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(img.cols, 64);
    EXPECT_EQ(img.rows, 64);
    EXPECT_EQ(c.Get("/v1/faces/12345/crop", key)->status, 404);
  }
  // image meta exposes identities
  {
    auto im = engine.get_image(7);
    auto r = c.Get("/v1/images/" + hvax::to_hex(im.sha256) + "/meta", key);
    ASSERT_EQ(r->status, 200);
    auto j = nlohmann::json::parse(r->body);
    EXPECT_EQ(j["identities"].size(), 2u);
    EXPECT_FALSE(j["repeat_identity"].get<bool>());
  }
  // curation: merge, split, rename, assign, dissolve
  {
    auto r = c.Post("/v1/identities/merge", key, nlohmann::json{{"ids", {id_a, id_b}}}.dump(), "application/json");
    ASSERT_EQ(r->status, 200) << r->body;
    auto j = nlohmann::json::parse(r->body);
    const int64_t target = j["identity"]["identity_id"].get<int64_t>();
    EXPECT_EQ(j["identity"]["size"], 8);
    EXPECT_TRUE(j["identity"]["pinned"].get<bool>());

    r = c.Post("/v1/identities/" + std::to_string(target) + "/split", key,
               nlohmann::json{{"face_ids", {1, 3, 5, 7}}}.dump(), "application/json");
    ASSERT_EQ(r->status, 200) << r->body;
    j = nlohmann::json::parse(r->body);
    EXPECT_EQ(j["identity"]["size"], 4);
    EXPECT_EQ(j["new_identity"]["size"], 4);
    const int64_t nid = j["new_identity"]["identity_id"].get<int64_t>();

    r = c.Patch("/v1/identities/" + std::to_string(nid), key, nlohmann::json{{"name", "person b"}}.dump(),
                "application/json");
    ASSERT_EQ(r->status, 200) << r->body;
    EXPECT_EQ(nlohmann::json::parse(r->body)["name"], "person b");

    r = c.Delete("/v1/identities/" + std::to_string(nid), key);
    EXPECT_EQ(r->status, 204);
    EXPECT_EQ(c.Get("/v1/identities/" + std::to_string(nid), key)->status, 404);
    EXPECT_EQ(nlohmann::json::parse(c.Get("/v1/faces/1", key)->body)["identity_id"].is_null(), true);
  }
  // evaluation and metrics
  {
    auto r = c.Get("/v1/eval/impostor?pairs=100", key);
    ASSERT_EQ(r->status, 200);
    auto j = nlohmann::json::parse(r->body);
    EXPECT_EQ(j["images_used"], 1);
    EXPECT_EQ(j["same_image_pairs"], 1);
    auto m = c.Get("/metrics");
    EXPECT_NE(m->body.find("hvax_identities "), std::string::npos);
    EXPECT_NE(m->body.find("hvax_faces_low_quality 1"), std::string::npos);
    EXPECT_NE(m->body.find("hvax_crop_total 1"), std::string::npos);
    auto plain = c.Get("/");
    EXPECT_NE(plain->body.find("identities"), std::string::npos);
  }
  std::filesystem::remove_all(dir);
}

TEST(HttpHelpers, PdfWhiteMarginsAreTrimmed) {
  cv::Mat page(160, 200, CV_8UC3, cv::Scalar(255, 255, 255));
  page(cv::Rect(50, 40, 100, 80)).setTo(cv::Scalar(20, 40, 80));
  page.at<cv::Vec3b>(2, 2) = cv::Vec3b(0, 0, 0);
  page.at<cv::Vec3b>(157, 197) = cv::Vec3b(0, 0, 0);
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(cv::imencode(".png", page, encoded));

  ASSERT_TRUE(hvax::trim_pdf_white_margins(encoded, 1'000'000));
  const cv::Mat cropped = cv::imdecode(encoded, cv::IMREAD_COLOR);
  ASSERT_FALSE(cropped.empty());
  EXPECT_LT(cropped.cols, page.cols);
  EXPECT_LT(cropped.rows, page.rows);
  EXPECT_GE(cropped.cols, 100);
  EXPECT_GE(cropped.rows, 80);
}

TEST(HttpHelpers, FullBleedPdfPageIsNotTrimmed) {
  cv::Mat page(80, 100, CV_8UC3, cv::Scalar(20, 40, 80));
  std::vector<uint8_t> encoded;
  ASSERT_TRUE(cv::imencode(".png", page, encoded));
  const auto original = encoded;

  EXPECT_FALSE(hvax::trim_pdf_white_margins(encoded, 1'000'000));
  EXPECT_EQ(encoded, original);
}

TEST(HttpApi, IdentityBrowsingIsOptIn) {
  auto dir = http_tmpdir();
  hvax::Config cfg;
  cfg.data_dir = dir.string();
  cfg.dedup = hvax::DedupMode::sha256;
  hvax::Engine engine(cfg);
  for (int i = 0; i < 3; ++i) ingest(engine, 10 + i, {face_at(person_embedding(1, i), 10, 10, 200)});
  ASSERT_TRUE(engine.recluster().has_value());
  LiveServer live(engine);
  auto c = live.client();
  // enumeration, curation, clustering and evaluation are refused ...
  EXPECT_EQ(c.Get("/v1/identities")->status, 403);
  EXPECT_EQ(c.Post("/v1/identities/cluster", "", "application/json")->status, 403);
  EXPECT_EQ(c.Post("/v1/identities/merge", "{\"ids\":[0,1]}", "application/json")->status, 403);
  EXPECT_EQ(c.Patch("/v1/identities/0", "{\"name\":\"x\"}", "application/json")->status, 403);
  EXPECT_EQ(c.Delete("/v1/identities/0")->status, 403);
  EXPECT_EQ(c.Get("/v1/eval/impostor?pairs=10")->status, 403);
  // ... while looking up a person reached from a hit still works
  const int64_t id = engine.get_face(0).identity_id;
  ASSERT_GE(id, 0);
  auto d = c.Get("/v1/identities/" + std::to_string(id));
  ASSERT_EQ(d->status, 200);
  EXPECT_EQ(nlohmann::json::parse(d->body)["size"], 3);
  EXPECT_EQ(c.Get("/v1/identities/" + std::to_string(id) + "/faces")->status, 200);
  EXPECT_EQ(c.Get("/v1/identities/" + std::to_string(id) + "/cooccurring")->status, 200);
  EXPECT_EQ(c.Get("/v1/identities/" + std::to_string(id) + "/timeline")->status, 200);
  EXPECT_EQ(c.Get("/v1/faces/0/crop?size=32")->status, 200);
  auto st = nlohmann::json::parse(c.Get("/v1/stats")->body);
  EXPECT_FALSE(st["identity_browse"].get<bool>());
  EXPECT_EQ(st["identities"], 1);
  auto plain = c.Get("/");
  EXPECT_EQ(plain->body.find("GET    /v1/identities\n"), std::string::npos);
  std::filesystem::remove_all(dir);
}

TEST(HttpApi, HidingRequiresTheIdentityKey) {
  auto dir = http_tmpdir();
  hvax::Config cfg;
  cfg.data_dir = dir.string();
  cfg.dedup = hvax::DedupMode::sha256;
  cfg.identity_key = "hush";
  hvax::Engine engine(cfg);
  for (int i = 0; i < 3; ++i) {
    ingest(engine, 10 + i, {face_at(person_embedding(1, i), 10, 10, 200)});
    ingest(engine, 20 + i, {face_at(person_embedding(2, i), 10, 10, 200)});
  }
  ingest(engine, 30, {face_at(person_embedding(1, 9), 10, 10, 200), face_at(person_embedding(2, 9), 300, 300, 200)});
  ASSERT_TRUE(engine.recluster().has_value());
  const int64_t id_b = engine.get_face(1).identity_id;
  ASSERT_GE(id_b, 0);
  LiveServer live(engine);
  auto c = live.client();
  const httplib::Headers key = {{"X-Identity-Key", "hush"}};
  const httplib::Headers generic = {{"X-API-Key", "hush"}};  // the landing page's key field

  // without the key: no hiding, no listing
  EXPECT_EQ(c.Post("/v1/identities/" + std::to_string(id_b) + "/hide", "", "application/json")->status, 401);
  EXPECT_EQ(c.Get("/v1/identities/hidden")->status, 401);
  EXPECT_FALSE(nlohmann::json::parse(c.Get("/v1/stats")->body)["identity_key_ok"].get<bool>());
  EXPECT_TRUE(nlohmann::json::parse(c.Get("/v1/stats", key)->body)["identity_key_ok"].get<bool>());
  // hide with the generic key slot
  auto r = c.Post("/v1/identities/" + std::to_string(id_b) + "/hide", generic, "", "application/json");
  ASSERT_EQ(r->status, 200) << r->body;
  EXPECT_TRUE(nlohmann::json::parse(r->body)["hidden"].get<bool>());
  auto hidden = nlohmann::json::parse(c.Get("/v1/identities/hidden", key)->body);
  ASSERT_EQ(hidden["identities"].size(), 1u);
  EXPECT_EQ(hidden["identities"][0]["identity_id"], id_b);

  // suppressed for the public ...
  EXPECT_EQ(c.Get("/v1/identities/" + std::to_string(id_b))->status, 404);
  EXPECT_EQ(c.Get("/v1/faces/1")->status, 404);
  EXPECT_EQ(c.Get("/v1/faces/1/crop?size=32")->status, 404);
  httplib::Headers h;
  h.emplace("X-K", "10");
  h.emplace("X-Min-Score", "-1");
  auto q = nlohmann::json::parse(c.Post("/v1/query/embedding", h, raw_embedding(person_embedding(2, 0)), "application/octet-stream")->body);
  for (auto& hit : q["hits"]) EXPECT_NE(hit["identity_id"], id_b);
  auto t = nlohmann::json::parse(c.Post("/v1/query/template", h, nlohmann::json{{"positive_embeddings", {std::vector<float>(person_embedding(2, 0).begin(), person_embedding(2, 0).end())}}}.dump(), "application/json")->body);
  for (auto& hit : t["hits"]) EXPECT_NE(hit["identity_id"], id_b);
  auto im = engine.get_image(7);
  auto meta = nlohmann::json::parse(c.Get("/v1/images/" + hvax::to_hex(im.sha256) + "/meta")->body);
  EXPECT_EQ(meta["face_ids"].size(), 1u);
  EXPECT_EQ(meta["identities"].size(), 1u);
  // ... visible and flagged for the admin
  EXPECT_EQ(c.Get("/v1/identities/" + std::to_string(id_b), key)->status, 200);
  EXPECT_EQ(c.Get("/v1/faces/1/crop?size=32", key)->status, 200);
  httplib::Headers hk = h;
  hk.emplace("X-Identity-Key", "hush");
  auto qa = nlohmann::json::parse(c.Post("/v1/query/embedding", hk, raw_embedding(person_embedding(2, 0)), "application/octet-stream")->body);
  bool saw = false;
  for (auto& hit : qa["hits"]) if (hit["identity_id"] == id_b) { saw = true; EXPECT_TRUE(hit["hidden"].get<bool>()); }
  EXPECT_TRUE(saw);
  auto ta = nlohmann::json::parse(c.Post("/v1/query/template", hk, nlohmann::json{{"positive_embeddings", {std::vector<float>(person_embedding(2, 0).begin(), person_embedding(2, 0).end())}}}.dump(), "application/json")->body);
  bool saw_t = false;
  for (auto& hit : ta["hits"]) if (hit["identity_id"] == id_b) { saw_t = true; EXPECT_TRUE(hit["hidden"].get<bool>()); }
  EXPECT_TRUE(saw_t) << "keyed template search shows hidden faces";
  EXPECT_EQ(nlohmann::json::parse(c.Get("/v1/images/" + hvax::to_hex(im.sha256) + "/meta", key)->body)["face_ids"].size(), 2u);
  EXPECT_NE(c.Get("/metrics")->body.find("hvax_identities_hidden 1"), std::string::npos);
  // unhide
  ASSERT_EQ(c.Post("/v1/identities/" + std::to_string(id_b) + "/unhide", key, "", "application/json")->status, 200);
  EXPECT_EQ(c.Get("/v1/identities/" + std::to_string(id_b))->status, 200);
  EXPECT_EQ(nlohmann::json::parse(c.Get("/v1/identities/hidden", key)->body)["total"], 0);
  std::filesystem::remove_all(dir);
}

TEST(HttpApi, HidingWithoutConfiguredKeyIsRefused) {
  auto dir = http_tmpdir();
  hvax::Config cfg;
  cfg.data_dir = dir.string();
  hvax::Engine engine(cfg);
  LiveServer live(engine);
  auto c = live.client();
  EXPECT_EQ(c.Post("/v1/identities/0/hide", httplib::Headers{{"X-Identity-Key", "anything"}}, "", "application/json")->status, 403);
  EXPECT_FALSE(nlohmann::json::parse(c.Get("/v1/stats")->body)["identity_key_configured"].get<bool>());
  std::filesystem::remove_all(dir);
}
