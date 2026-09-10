// Identity layer: quality gate, int8 file upgrade, incremental assignment,
// clustering, curation, co-occurrence, timeline, reindex, range search.
#include "hvax/align/umeyama.hpp"
#include "hvax/embed/arcface.hpp"
#include "hvax/identity/cluster.hpp"
#include "hvax/store/gallery.hpp"
#include "hvax/util/sha256.hpp"

#include <atomic>
#include <filesystem>
#include <random>
#include <unistd.h>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

std::filesystem::path tmpdir() {
  static std::atomic<int> seq{0};
  auto p = std::filesystem::temp_directory_path() /
           ("hvax-identity-test-" + std::to_string(::getpid()) + "-" + std::to_string(seq.fetch_add(1)));
  std::filesystem::create_directories(p);
  return p;
}

cv::Mat photo(int seed, int w, int h) {
  cv::Mat img(h, w, CV_8UC3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      img.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uint8_t>(x + seed), static_cast<uint8_t>(y + seed * 2), 80);
  return img;
}

std::vector<uint8_t> encode_jpg(const cv::Mat& im) {
  std::vector<uint8_t> buf;
  cv::imencode(".jpg", im, buf, {cv::IMWRITE_JPEG_QUALITY, 90});
  return buf;
}

// Deterministic "person": a fixed random direction plus per-face noise.
hvax::Embedding person_embedding(int person, int sample, float noise = 0.35f) {
  std::mt19937 base(1000 + person);
  std::normal_distribution<float> nd(0.f, 1.f);
  hvax::Embedding e{};
  for (auto& x : e) x = nd(base);
  std::mt19937 rng(static_cast<unsigned>(person * 100003 + sample));
  for (auto& x : e) x += noise * nd(rng);
  hvax::l2_normalize(e.data());
  return e;
}

hvax::DetectedFace face_at(const hvax::Embedding& e, float x, float y, float size, float det = 0.9f) {
  hvax::DetectedFace f;
  f.box = {x, y, x + size, y + size};
  f.det_score = det;
  f.kps = hvax::arcface_dst();
  f.embedding = e;
  return f;
}

struct Inserted {
  int64_t image_id;
  std::vector<uint8_t> bytes;
};

Inserted insert_image(hvax::Gallery& g, int seed, const std::vector<hvax::DetectedFace>& faces, int w = 640,
                      int h = 640) {
  auto img = photo(seed, w, h);
  auto bytes = encode_jpg(img);
  auto id = g.insert(bytes, img, hvax::sha256_bytes(bytes), hvax::Mime::jpeg, hvax::hash_image(img), faces);
  return {id, bytes};
}

hvax::Config make_cfg(const std::filesystem::path& dir) {
  hvax::Config cfg;
  cfg.data_dir = dir.string();
  return cfg;
}

}  // namespace

TEST(QualityGate, TinyFacesStoredButNotIndexed) {
  auto dir = tmpdir();
  hvax::Gallery g(make_cfg(dir));
  auto big = person_embedding(1, 0);
  auto tiny = person_embedding(2, 0);
  insert_image(g, 1, {face_at(big, 10, 10, 200), face_at(tiny, 300, 300, 10)});
  EXPECT_EQ(g.live_faces(), 2u);
  EXPECT_EQ(g.indexed_faces(), 1u);
  EXPECT_EQ(g.low_quality_faces(), 1u);
  auto f1 = g.face(1);
  EXPECT_TRUE(f1.flags & hvax::kLowQuality);
  EXPECT_LT(f1.quality, 0.05f);
  EXPECT_GT(g.face(0).quality, 0.8f);

  auto hidden = g.search(tiny.data(), 5, 0.5f);
  EXPECT_TRUE(hidden.empty());
  hvax::SearchOptions o;
  o.k = 5;
  o.min_score = 0.5f;
  o.include_low_quality = true;
  auto shown = g.search(tiny.data(), o);
  ASSERT_EQ(shown.size(), 1u);
  EXPECT_EQ(shown[0].face_id, 1);
  EXPECT_TRUE(shown[0].flags & hvax::kLowQuality);
  std::filesystem::remove_all(dir);
}

TEST(Int8File, LegacyScaleIsRequantizedOnOpen) {
  auto dir = tmpdir();
  auto e = person_embedding(3, 0);
  {
    hvax::Gallery g(make_cfg(dir));
    insert_image(g, 1, {face_at(e, 10, 10, 200)});
  }
  {
    // Rewrite the int8 file the way the old code did: version 1, scale 127.
    hvax::SlotFile<hvax::EmbI8> f;
    f.open(dir / "embeddings.i8", "HVAXEI81");
    ASSERT_EQ(f.size(), 1u);
    for (int i = 0; i < hvax::kDim; ++i) f.at(0).v[i] = static_cast<int8_t>(std::lrintf(e[static_cast<size_t>(i)] * 127.f));
    f.set_version(1);
    f.fsync_all();
  }
  {
    hvax::Gallery g(make_cfg(dir));
    EXPECT_EQ(g.live_faces(), 1u);
  }
  hvax::SlotFile<hvax::EmbI8> f;
  f.open(dir / "embeddings.i8", "HVAXEI81");
  EXPECT_EQ(f.version(), hvax::kI8FileVersion);
  std::vector<int8_t> want(hvax::kDim);
  hvax::quantize_i8(e.data(), want.data());
  for (int i = 0; i < hvax::kDim; ++i) EXPECT_EQ(f.at(0).v[i], want[static_cast<size_t>(i)]);
  std::filesystem::remove_all(dir);
}

namespace {

// Two people, three solo photos each, plus one photo with both.
struct TwoPeople {
  std::vector<int64_t> images_a, images_b;
  int64_t image_ab = 0;
};

TwoPeople seed_two_people(hvax::Gallery& g) {
  TwoPeople t;
  for (int i = 0; i < 3; ++i) {
    t.images_a.push_back(insert_image(g, 10 + i, {face_at(person_embedding(1, i), 10, 10, 200)}).image_id);
    t.images_b.push_back(insert_image(g, 20 + i, {face_at(person_embedding(2, i), 10, 10, 200)}).image_id);
  }
  t.image_ab = insert_image(g, 30, {face_at(person_embedding(1, 9), 10, 10, 200), face_at(person_embedding(2, 9), 300, 300, 200)}).image_id;
  return t;
}

void check_identity_layer(hvax::Config cfg) {
  auto dir = std::filesystem::path(cfg.data_dir);
  hvax::Gallery g(cfg);
  seed_two_people(g);
  EXPECT_EQ(g.identity_count(), 0u);
  EXPECT_EQ(g.unassigned_faces(), 8u);

  hvax::ClusterParams p;
  p.threads = 2;
  auto rep = g.recluster(p);
  ASSERT_TRUE(rep.has_value());
  EXPECT_EQ(rep->faces_eligible, 8u);
  EXPECT_EQ(rep->identities, 2u);
  EXPECT_EQ(rep->faces_assigned, 8u);
  EXPECT_EQ(rep->largest, 4u);
  EXPECT_EQ(g.identity_count(), 2u);
  EXPECT_EQ(g.unassigned_faces(), 0u);

  const int64_t id_a = g.face(0).identity_id;
  const int64_t id_b = g.face(1).identity_id;
  ASSERT_GE(id_a, 0);
  ASSERT_GE(id_b, 0);
  EXPECT_NE(id_a, id_b);
  for (int i = 0; i < 6; i += 2) EXPECT_EQ(g.face(i).identity_id, id_a);
  for (int i = 1; i < 6; i += 2) EXPECT_EQ(g.face(i).identity_id, id_b);

  auto va = g.identity(id_a);
  ASSERT_TRUE(va.has_value());
  EXPECT_EQ(va->size, 4u);
  EXPECT_EQ(va->n_images, 4u);
  EXPECT_GT(va->cohesion, 0.8f);
  EXPECT_GE(va->rep_face_id, 0);
  EXPECT_EQ(g.face(va->rep_face_id).identity_id, id_a);

  // incremental join at ingest
  insert_image(g, 40, {face_at(person_embedding(1, 50), 10, 10, 200)});
  EXPECT_EQ(g.face(8).identity_id, id_a);
  EXPECT_EQ(g.identity(id_a)->size, 5u);
  // a stranger stays unassigned
  insert_image(g, 41, {face_at(person_embedding(3, 0), 10, 10, 200)});
  EXPECT_EQ(g.face(9).identity_id, -1);
  EXPECT_EQ(g.unassigned_faces(), 1u);

  // identity search
  auto ih = g.search_identities(person_embedding(1, 77).data(), 2, -1.f);
  ASSERT_EQ(ih.size(), 2u);
  EXPECT_EQ(ih[0].identity_id, id_a);
  EXPECT_GT(ih[0].score, ih[1].score);
  ASSERT_TRUE(ih[0].best_face.has_value());
  EXPECT_EQ(ih[0].best_face->identity_id, id_a);

  // grouped search: many faces of A collapse into one hit
  hvax::SearchOptions go;
  go.k = 5;
  go.min_score = 0.f;
  go.group_by_identity = true;
  auto grouped = g.search(person_embedding(1, 78).data(), go);
  ASSERT_GE(grouped.size(), 2u);
  EXPECT_EQ(grouped[0].identity_id, id_a);
  EXPECT_EQ(grouped[0].collapsed, 4);
  int seen_a = 0;
  for (auto& h : grouped) seen_a += h.identity_id == id_a;
  EXPECT_EQ(seen_a, 1);

  // range search
  hvax::SearchOptions ro;
  ro.k = 1;
  ro.min_score = 0.5f;
  ro.range = true;
  auto ranged = g.search(person_embedding(1, 79).data(), ro);
  EXPECT_EQ(ranged.size(), 5u);
  for (auto& h : ranged) EXPECT_EQ(h.identity_id, id_a);

  // co-occurrence and timeline
  auto co = g.cooccurring(id_a, 10);
  ASSERT_EQ(co.size(), 1u);
  EXPECT_EQ(co[0].identity_id, id_b);
  EXPECT_EQ(co[0].shared_images, 1u);
  auto tl = g.timeline(id_a, 86400000);
  uint32_t total = 0;
  for (auto& b : tl) total += b.faces;
  EXPECT_EQ(total, 5u);
  auto ids = g.identities_of(7);
  ASSERT_EQ(ids.size(), 2u);

  // faces listing sorted by cosine to centroid
  std::vector<float> scores;
  auto faces = g.identity_faces(id_a, hvax::FaceSort::score, 0, 100, &scores);
  ASSERT_EQ(faces.size(), 5u);
  for (size_t i = 1; i < scores.size(); ++i) EXPECT_GE(scores[i - 1], scores[i]);

  // curation: merge pins; recluster must keep the pinned merge
  auto merged = g.merge_identities({id_a, id_b});
  ASSERT_TRUE(merged.has_value());
  EXPECT_EQ(g.identity_count(), 1u);
  EXPECT_EQ(g.identity(*merged)->size, 9u);
  EXPECT_TRUE(g.identity(*merged)->flags & hvax::kIdentityPinned);
  auto rep2 = g.recluster(p);
  ASSERT_TRUE(rep2.has_value());
  EXPECT_EQ(rep2->identities_pinned, 1u);
  EXPECT_EQ(g.identity(*merged)->size, 9u) << "pinned faces must not be re-split by clustering";
  // the stranger with no near neighbours forms no cluster
  EXPECT_EQ(g.face(9).identity_id, -1);

  // split back out person B's faces; both halves are pinned and stay apart
  auto nid = g.split_identity(*merged, {1, 3, 5, 7});
  ASSERT_TRUE(nid.has_value());
  EXPECT_EQ(g.identity(*merged)->size, 5u);
  EXPECT_EQ(g.identity(*nid)->size, 4u);
  auto rep3 = g.recluster(p);
  ASSERT_TRUE(rep3.has_value());
  EXPECT_EQ(g.identity_count(), 2u);
  EXPECT_EQ(g.identity(*nid)->size, 4u);
  EXPECT_TRUE(g.rename_identity(*nid, "person b"));
  EXPECT_EQ(g.identity(*nid)->name, "person b");

  // manual assignment of the stranger, then dissolve
  EXPECT_TRUE(g.assign_face(9, *nid));
  EXPECT_EQ(g.identity(*nid)->size, 5u);
  EXPECT_TRUE(g.dissolve_identity(*nid));
  EXPECT_FALSE(g.identity(*nid).has_value());
  EXPECT_EQ(g.face(1).identity_id, -1);
  EXPECT_EQ(g.identity_count(), 1u);

  // deleting an image removes its faces from the identity
  const auto sha_img0 = g.image(1).sha256;
  (void)sha_img0;
  EXPECT_TRUE(g.remove_image(1));
  EXPECT_EQ(g.identity(*merged)->size, 4u);

  // persistence: reopen and everything is still there
  const uint64_t before = g.identity_count();
  const int64_t face8_identity = g.face(8).identity_id;
  g.flush();
  {
    hvax::Gallery g2(cfg);
    EXPECT_EQ(g2.identity_count(), before);
    EXPECT_EQ(g2.face(8).identity_id, face8_identity);
    EXPECT_EQ(g2.identity(*merged)->size, 4u);
    EXPECT_EQ(g2.identity(*merged)->name, "");
  }
  std::filesystem::remove_all(dir);
}

}  // namespace

TEST(Identity, LayerOnExactTier) {
  auto dir = tmpdir();
  auto cfg = make_cfg(dir);
  cfg.exact_until = 100000;
  check_identity_layer(cfg);
}

TEST(Identity, LayerOnHnswTier) {
  auto dir = tmpdir();
  auto cfg = make_cfg(dir);
  cfg.exact_until = 1;  // every search and the kNN graph go through usearch
  check_identity_layer(cfg);
}

TEST(Identity, LayerWithoutInt8Scan) {
  auto dir = tmpdir();
  auto cfg = make_cfg(dir);
  cfg.i8_scan = false;
  check_identity_layer(cfg);
}

TEST(Identity, ReindexAppliesNewGate) {
  auto dir = tmpdir();
  auto cfg = make_cfg(dir);
  {
    hvax::Gallery g(cfg);
    seed_two_people(g);
    hvax::ClusterParams p;
    p.threads = 2;
    ASSERT_TRUE(g.recluster(p).has_value());
    EXPECT_EQ(g.identity_count(), 2u);
    EXPECT_EQ(g.indexed_faces(), 8u);
  }
  cfg.index_min_face_px = 300;  // stricter than every 200 px face
  cfg.reindex = true;
  hvax::Gallery g(cfg);
  auto r = g.reindex();
  EXPECT_EQ(r.rows, 8u);
  EXPECT_EQ(r.low_quality, 8u);
  EXPECT_EQ(r.indexed, 0u);
  EXPECT_EQ(r.unassigned_by_gate, 8u);
  EXPECT_EQ(r.requantized, 8u);
  EXPECT_EQ(g.identity_count(), 0u) << "identities with no eligible faces are tombstoned";
  EXPECT_EQ(g.low_quality_faces(), 8u);
  std::filesystem::remove_all(dir);
}

TEST(Hnsw, SearchWorksOnIndexLoadedFromDisk) {
  // Regression: a restarted server used to crash on its first HNSW search
  // because usearch::load() does not allocate per-thread search contexts.
  auto dir = tmpdir();
  auto cfg = make_cfg(dir);
  cfg.exact_until = 1;
  {
    hvax::Gallery g(cfg);
    seed_two_people(g);
    EXPECT_EQ(g.indexed_faces(), 8u);
    g.flush();
  }
  hvax::Gallery g(cfg);
  EXPECT_EQ(g.indexed_faces(), 8u);
  auto hits = g.search(person_embedding(1, 3).data(), 5, 0.5f);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits[0].face_id % 2, 0) << "person 1 lives on even face ids";
  hvax::SearchOptions ro;
  ro.k = 1;
  ro.min_score = 0.5f;
  ro.range = true;
  EXPECT_EQ(g.search(person_embedding(1, 4).data(), ro).size(), 4u);
  // clustering right after a restart also goes through the loaded index
  hvax::ClusterParams p;
  p.threads = 4;
  auto rep = g.recluster(p);
  ASSERT_TRUE(rep.has_value());
  EXPECT_EQ(rep->identities, 2u);
  std::filesystem::remove_all(dir);
}

TEST(Identity, UpgradeKeepsIdentity) {
  auto dir = tmpdir();
  hvax::Gallery g(make_cfg(dir));
  seed_two_people(g);
  hvax::ClusterParams p;
  p.threads = 2;
  ASSERT_TRUE(g.recluster(p).has_value());
  const int64_t id_a = g.face(0).identity_id;
  auto big = photo(99, 1280, 1280);
  auto bytes = encode_jpg(big);
  auto faces = g.upgrade(1, bytes, big, hvax::sha256_bytes(bytes), hvax::Mime::jpeg, hvax::hash_image(big),
                         {face_at(person_embedding(1, 0, 0.36f), 20, 20, 400)});
  ASSERT_EQ(faces.size(), 1u);
  EXPECT_EQ(faces[0].face_id, 0);
  EXPECT_EQ(faces[0].identity_id, id_a);
  EXPECT_EQ(g.identity(id_a)->size, 4u);
  std::filesystem::remove_all(dir);
}

TEST(Identity, ImpostorEvalUsesSameImageNegatives) {
  auto dir = tmpdir();
  hvax::Gallery g(make_cfg(dir));
  // ten photos with two different people each
  for (int i = 0; i < 10; ++i)
    insert_image(g, 100 + i, {face_at(person_embedding(10 + i, 0), 10, 10, 200), face_at(person_embedding(40 + i, 0), 300, 300, 200)});
  auto r = g.impostor_eval(1000, 5);
  EXPECT_EQ(r.images_used, 10u);
  EXPECT_EQ(r.same_image_pairs, 10u);
  EXPECT_GT(r.random_pairs, 0u);
  EXPECT_LT(std::abs(r.same_image_mean), 0.2);
  EXPECT_EQ(r.same_image_far.size(), r.random_above.size());
  std::filesystem::remove_all(dir);
}

TEST(ChineseWhispers, SeparatesTwoCliquesAndRespectsFixedLabels) {
  // nodes 0-3 form a clique, 4-7 form a clique, one weak bridge 3-4
  hvax::KnnGraph g;
  g.n = 8;
  g.k = 4;
  g.nbr.assign(32, UINT32_MAX);
  g.sim.assign(32, -2.f);
  auto set = [&](uint32_t u, int slot, uint32_t v, float w) {
    g.nbr[u * 4 + slot] = v;
    g.sim[u * 4 + slot] = w;
  };
  for (uint32_t u = 0; u < 4; ++u) {
    int s = 0;
    for (uint32_t v = 0; v < 4; ++v)
      if (v != u) set(u, s++, v, 0.9f);
  }
  for (uint32_t u = 4; u < 8; ++u) {
    int s = 0;
    for (uint32_t v = 4; v < 8; ++v)
      if (v != u) set(u, s++, v, 0.9f);
  }
  set(3, 3, 4, 0.6f);
  std::vector<uint8_t> eligible(8, 1);
  auto csr = hvax::symmetrize_knn(g, 0.55f, eligible);
  EXPECT_EQ(csr.edges(), 13u);
  std::vector<uint32_t> labels(8);
  for (uint32_t i = 0; i < 8; ++i) labels[i] = i;
  std::vector<uint8_t> fixed(8, 0);
  hvax::chinese_whispers(csr, labels, fixed, 20, 2);
  EXPECT_EQ(hvax::compact_labels(labels), 2u);
  EXPECT_EQ(labels[0], labels[3]);
  EXPECT_EQ(labels[4], labels[7]);
  EXPECT_NE(labels[0], labels[4]);

  // pin node 5 with a label of its own: it never changes, its clique adopts
  // the pinned label, and the first clique still ends up elsewhere
  for (uint32_t i = 0; i < 8; ++i) labels[i] = i;
  labels[5] = 100;
  fixed[5] = 1;
  hvax::chinese_whispers(csr, labels, fixed, 20, 2);
  EXPECT_EQ(labels[5], 100u);
  // with exactly tied weights the smallest label wins, so the rest of clique
  // two settles on a label of its own rather than the pinned one
  EXPECT_EQ(labels[4], labels[6]);
  EXPECT_EQ(labels[6], labels[7]);
  EXPECT_EQ(labels[0], labels[3]);
  EXPECT_NE(labels[0], labels[4]);
  EXPECT_NE(labels[0], 100u);
}

TEST(Identity, HiddenIdentitiesAreSuppressedEverywhere) {
  auto dir = tmpdir();
  hvax::Gallery g(make_cfg(dir));
  seed_two_people(g);
  hvax::ClusterParams p;
  p.threads = 2;
  ASSERT_TRUE(g.recluster(p).has_value());
  const int64_t id_a = g.face(0).identity_id;
  const int64_t id_b = g.face(1).identity_id;
  ASSERT_TRUE(g.set_identity_hidden(id_b, true));
  EXPECT_TRUE(g.identity_hidden(id_b));
  EXPECT_TRUE(g.face_hidden(1));
  EXPECT_FALSE(g.face_hidden(0));
  EXPECT_EQ(g.hidden_identity_count(), 1u);
  ASSERT_EQ(g.hidden_identities().size(), 1u);
  EXPECT_EQ(g.hidden_identities()[0].identity_id, id_b);
  EXPECT_TRUE(g.hidden_identities()[0].flags & hvax::kIdentityPinned) << "hiding pins";

  // face search: B's faces vanish, even when the query is one of them
  auto hits = g.search(person_embedding(2, 0).data(), 10, -1.f);
  for (auto& h : hits) EXPECT_NE(h.identity_id, id_b);
  hvax::SearchOptions o;
  o.k = 10;
  o.min_score = -1.f;
  o.include_hidden = true;
  auto admin_hits = g.search(person_embedding(2, 0).data(), o);
  EXPECT_TRUE(std::any_of(admin_hits.begin(), admin_hits.end(), [&](auto& h) { return h.identity_id == id_b && h.hidden; }));
  // identity search, lookups, listing, co-occurrence
  auto ih = g.search_identities(person_embedding(2, 1).data(), 5, -1.f);
  for (auto& h : ih) EXPECT_NE(h.identity_id, id_b);
  EXPECT_FALSE(g.identity(id_b).has_value());
  EXPECT_TRUE(g.identity(id_b, true).has_value());
  EXPECT_TRUE(g.identity_faces(id_b, hvax::FaceSort::score, 0, 10).empty());
  EXPECT_EQ(g.identity_faces(id_b, hvax::FaceSort::score, 0, 10, nullptr, true).size(), 4u);
  EXPECT_EQ(g.list_identities(hvax::IdentitySort::size, 0, 10).size(), 1u);
  EXPECT_EQ(g.list_identities(hvax::IdentitySort::size, 0, 10, true).size(), 2u);
  EXPECT_TRUE(g.cooccurring(id_a, 10).empty()) << "the hidden co-star is not reported";
  EXPECT_EQ(g.cooccurring(id_a, 10, true).size(), 1u);
  // template search skips hidden faces too
  std::array<hvax::Embedding, 1> pos{person_embedding(2, 2)};
  for (auto& h : g.search_template(pos, {}, {}, 10, -1.f)) EXPECT_NE(h.identity_id, id_b);
  auto tpl_admin = g.search_template(pos, {}, {}, 10, -1.f, true);
  EXPECT_TRUE(std::any_of(tpl_admin.begin(), tpl_admin.end(), [&](auto& h) { return h.identity_id == id_b && h.hidden; }));
  // a new face of the hidden person joins it and is suppressed immediately
  insert_image(g, 60, {face_at(person_embedding(2, 40), 10, 10, 200)});
  EXPECT_EQ(g.face(8).identity_id, id_b);
  EXPECT_TRUE(g.face_hidden(8));
  // survives a recluster and a reopen
  ASSERT_TRUE(g.recluster(p).has_value());
  EXPECT_TRUE(g.identity_hidden(id_b));
  g.flush();
  {
    hvax::Gallery g2(make_cfg(dir));
    EXPECT_TRUE(g2.identity_hidden(id_b));
    EXPECT_EQ(g2.hidden_identity_count(), 1u);
    EXPECT_TRUE(g2.search(person_embedding(2, 0).data(), 10, -1.f).empty() ||
                g2.search(person_embedding(2, 0).data(), 10, -1.f)[0].identity_id != id_b);
  }
  // unhide restores everything
  ASSERT_TRUE(g.set_identity_hidden(id_b, false));
  EXPECT_EQ(g.hidden_identity_count(), 0u);
  EXPECT_TRUE(g.identity(id_b).has_value());
  EXPECT_FALSE(g.cooccurring(id_a, 10).empty());
  std::filesystem::remove_all(dir);
}
