// Identity layer of the Gallery: assignment, curation, batch clustering and
// the label-free impostor evaluation. Everything here operates on the same
// mmaps as gallery.cpp and follows its locking rules.
#include "hvax/embed/arcface.hpp"
#include "hvax/identity/cluster.hpp"
#include "hvax/store/gallery.hpp"
#include "hvax/util/time.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <spdlog/spdlog.h>

namespace hvax {
namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

float iou(const FaceSlot& a, const FaceSlot& b) {
  const float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.f, xx2 - xx1) * std::max(0.f, yy2 - yy1);
  const float area = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return area <= 0.f ? 0.f : inter / area;
}

}  // namespace

// ---------------------------------------------------------------------------
// state

bool Gallery::identity_live(int64_t id) const {
  return id >= 0 && static_cast<uint64_t>(id) < identities_.size() && slot_live(identities_.at(static_cast<uint64_t>(id)).flags);
}

void Gallery::refresh_centroid_norm_locked(int64_t id) {
  const size_t base = static_cast<size_t>(id) * kDim;
  if (centroid_norm_.size() < base + kDim) centroid_norm_.resize(base + kDim, 0.f);
  float* dst = centroid_norm_.data() + base;
  std::memcpy(dst, centroids_.at(static_cast<uint64_t>(id)).v, sizeof(float) * kDim);
  double n2 = 0;
  for (int i = 0; i < kDim; ++i) n2 += static_cast<double>(dst[i]) * dst[i];
  if (n2 > 0) l2_normalize(dst);
}

void Gallery::load_identity_state_locked() {
  while (centroids_.size() < identities_.size()) centroids_.append(EmbF32{});
  const uint64_t C = identities_.size();
  members_.assign(static_cast<size_t>(C), {});
  centroid_norm_.assign(static_cast<size_t>(C) * kDim, 0.f);
  const uint64_t n = std::min(embs_.size(), faces_.size());
  for (uint64_t i = 0; i < n; ++i) {
    FaceSlot& f = faces_.at(i);
    if (!slot_live(f.flags) || f.identity_ref == 0) continue;
    const int64_t id = identity_of(f);
    if (!identity_live(id) || !face_indexable(f.flags)) {
      f.identity_ref = 0;  // dangling reference or gated out: repair
      continue;
    }
    members_[static_cast<size_t>(id)].push_back(static_cast<uint32_t>(i));
  }
  const int64_t now = unix_ms();
  for (uint64_t id = 0; id < C; ++id) {
    auto& s = identities_.at(id);
    if (!slot_live(s.flags)) continue;
    if (s.size != members_[static_cast<size_t>(id)].size()) recompute_identity_locked(static_cast<int64_t>(id), now);
    else refresh_centroid_norm_locked(static_cast<int64_t>(id));
  }
}

int64_t Gallery::new_identity_locked(int64_t now) {
  IdentitySlot s;
  s.identity_id = identities_.size();
  s.created_at = static_cast<uint64_t>(now);
  s.updated_at = static_cast<uint64_t>(now);
  s.flags = 0;
  const uint64_t id = identities_.append(s);
  centroids_.append(EmbF32{});
  members_.emplace_back();
  centroid_norm_.resize(static_cast<size_t>(id + 1) * kDim, 0.f);
  return static_cast<int64_t>(id);
}

void Gallery::recompute_identity_locked(int64_t id, int64_t now) {
  if (id < 0 || static_cast<uint64_t>(id) >= identities_.size()) return;
  IdentitySlot& s = identities_.at(static_cast<uint64_t>(id));
  auto& rows = members_[static_cast<size_t>(id)];
  EmbF32& c = centroids_.at(static_cast<uint64_t>(id));
  if (rows.empty()) {
    s.size = 0;
    s.n_images = 0;
    s.rep_face_id = 0;
    s.flags |= kTombstone;
    s.updated_at = static_cast<uint64_t>(now);
    std::memset(c.v, 0, sizeof(c.v));
    refresh_centroid_norm_locked(id);
    return;
  }
  std::vector<double> acc(kDim, 0.0);
  uint64_t first = UINT64_MAX, last = 0;
  std::vector<uint64_t> images;
  images.reserve(rows.size());
  for (uint32_t r : rows) {
    const float* x = embs_.at(r).v;
    for (int i = 0; i < kDim; ++i) acc[static_cast<size_t>(i)] += x[i];
    const auto& f = faces_.at(r);
    first = std::min(first, f.created_at);
    last = std::max(last, f.created_at);
    images.push_back(f.image_id);
  }
  for (int i = 0; i < kDim; ++i) c.v[i] = static_cast<float>(acc[static_cast<size_t>(i)] / static_cast<double>(rows.size()));
  refresh_centroid_norm_locked(id);
  const float* cn = centroid_norm_.data() + static_cast<size_t>(id) * kDim;
  float best = -2.f;
  uint32_t rep = rows.front();
  for (uint32_t r : rows) {
    const float sc = dot512(cn, embs_.at(r).v);
    if (sc > best) {
      best = sc;
      rep = r;
    }
  }
  std::sort(images.begin(), images.end());
  s.size = static_cast<uint32_t>(rows.size());
  s.n_images = static_cast<uint32_t>(std::unique(images.begin(), images.end()) - images.begin());
  s.rep_face_id = rep;
  s.first_seen = first;
  s.last_seen = last;
  s.updated_at = static_cast<uint64_t>(now);
  s.flags &= ~kTombstone;
}

void Gallery::assign_locked(uint64_t row, int64_t id, bool touch_stats) {
  FaceSlot& f = faces_.at(row);
  if (identity_of(f) == id) return;
  if (f.identity_ref != 0) unassign_locked(row);
  f.identity_ref = static_cast<uint64_t>(id) + 1;
  members_[static_cast<size_t>(id)].push_back(static_cast<uint32_t>(row));
  if (touch_stats) recompute_identity_locked(id, unix_ms());
}

void Gallery::unassign_locked(uint64_t row) {
  FaceSlot& f = faces_.at(row);
  const int64_t id = identity_of(f);
  if (id < 0) return;
  f.identity_ref = 0;
  if (static_cast<uint64_t>(id) >= members_.size()) return;
  auto& rows = members_[static_cast<size_t>(id)];
  rows.erase(std::remove(rows.begin(), rows.end(), static_cast<uint32_t>(row)), rows.end());
  recompute_identity_locked(id, unix_ms());
}

int64_t Gallery::best_identity_locked(const float* query, float* score_out) const {
  int64_t best = -1;
  float best_s = -2.f;
  const uint64_t C = identities_.size();
  for (uint64_t id = 0; id < C; ++id) {
    if (!slot_live(identities_.at(id).flags)) continue;
    const float s = dot512(query, centroid_norm_.data() + static_cast<size_t>(id) * kDim);
    if (s > best_s) {
      best_s = s;
      best = static_cast<int64_t>(id);
    }
  }
  if (score_out) *score_out = best_s;
  return best;
}

void Gallery::try_join_identity_locked(uint64_t row) {
  if (identities_.size() == 0) return;
  float s = -2.f;
  const int64_t id = best_identity_locked(embs_.at(row).v, &s);
  if (id >= 0 && s >= cfg_.identity_join) assign_locked(row, id, true);
}

// ---------------------------------------------------------------------------
// queries

std::vector<IdentityHit> Gallery::search_identities(const float* query, int k, float min_score) const {
  std::shared_lock lock(mu_);
  std::vector<IdentityHit> out;
  if (k <= 0) return out;
  struct Item {
    float s;
    int64_t id;
  };
  auto cmp = [](const Item& a, const Item& b) { return a.s > b.s; };
  std::priority_queue<Item, std::vector<Item>, decltype(cmp)> heap(cmp);
  const uint64_t C = identities_.size();
  for (uint64_t id = 0; id < C; ++id) {
    if (!slot_live(identities_.at(id).flags)) continue;
    const float s = dot512(query, centroid_norm_.data() + static_cast<size_t>(id) * kDim);
    if (s < min_score) continue;
    if (static_cast<int>(heap.size()) < k) heap.push(Item{s, static_cast<int64_t>(id)});
    else if (s > heap.top().s) {
      heap.pop();
      heap.push(Item{s, static_cast<int64_t>(id)});
    }
  }
  std::vector<Item> items;
  while (!heap.empty()) {
    items.push_back(heap.top());
    heap.pop();
  }
  std::sort(items.begin(), items.end(), [](auto& a, auto& b) { return a.s > b.s; });
  for (const auto& it : items) {
    const auto& s = identities_.at(static_cast<uint64_t>(it.id));
    IdentityHit h;
    h.identity_id = it.id;
    h.score = it.s;
    h.size = s.size;
    h.n_images = s.n_images;
    h.rep_face_id = static_cast<int64_t>(s.rep_face_id);
    h.name.assign(s.name, strnlen(s.name, kIdentityNameBytes));
    float best = -2.f;
    uint32_t best_row = UINT32_MAX;
    for (uint32_t r : members_[static_cast<size_t>(it.id)]) {
      const float sc = dot512(query, embs_.at(r).v);
      if (sc > best) {
        best = sc;
        best_row = r;
      }
    }
    if (best_row != UINT32_MAX) {
      bool ok = false;
      Hit bf = hydrate_one(best_row, best, ok);
      if (ok) h.best_face = bf;
    }
    out.push_back(std::move(h));
  }
  return out;
}

std::optional<IdentityView> Gallery::identity(int64_t identity_id) const {
  std::shared_lock lock(mu_);
  if (!identity_live(identity_id)) return std::nullopt;
  auto v = identity_from_slot(identities_.at(static_cast<uint64_t>(identity_id)));
  const auto& rows = members_[static_cast<size_t>(identity_id)];
  if (!rows.empty()) {
    const float* cn = centroid_norm_.data() + static_cast<size_t>(identity_id) * kDim;
    double acc = 0;
    for (uint32_t r : rows) acc += dot512(cn, embs_.at(r).v);
    v.cohesion = static_cast<float>(acc / static_cast<double>(rows.size()));
  }
  return v;
}

std::vector<IdentityView> Gallery::list_identities(IdentitySort sort, uint64_t offset, uint64_t limit) const {
  std::shared_lock lock(mu_);
  std::vector<uint64_t> ids;
  const uint64_t C = identities_.size();
  ids.reserve(static_cast<size_t>(C));
  for (uint64_t id = 0; id < C; ++id)
    if (slot_live(identities_.at(id).flags)) ids.push_back(id);
  auto key_size = [&](uint64_t a, uint64_t b) {
    const auto& x = identities_.at(a);
    const auto& y = identities_.at(b);
    return x.size != y.size ? x.size > y.size : a < b;
  };
  auto key_recent = [&](uint64_t a, uint64_t b) {
    const auto& x = identities_.at(a);
    const auto& y = identities_.at(b);
    return x.last_seen != y.last_seen ? x.last_seen > y.last_seen : a < b;
  };
  if (sort == IdentitySort::size) std::sort(ids.begin(), ids.end(), key_size);
  else if (sort == IdentitySort::recent) std::sort(ids.begin(), ids.end(), key_recent);
  std::vector<IdentityView> out;
  for (uint64_t i = offset; i < ids.size() && out.size() < limit; ++i) {
    const uint64_t id = ids[static_cast<size_t>(i)];
    auto v = identity_from_slot(identities_.at(id));
    const auto& rows = members_[static_cast<size_t>(id)];
    if (!rows.empty()) {
      const float* cn = centroid_norm_.data() + static_cast<size_t>(id) * kDim;
      double acc = 0;
      for (uint32_t r : rows) acc += dot512(cn, embs_.at(r).v);
      v.cohesion = static_cast<float>(acc / static_cast<double>(rows.size()));
    }
    out.push_back(std::move(v));
  }
  return out;
}

std::vector<FaceView> Gallery::identity_faces(int64_t identity_id, FaceSort sort, uint64_t offset, uint64_t limit,
                                              std::vector<float>* scores) const {
  std::shared_lock lock(mu_);
  std::vector<FaceView> out;
  if (!identity_live(identity_id)) return out;
  const auto& rows = members_[static_cast<size_t>(identity_id)];
  const float* cn = centroid_norm_.data() + static_cast<size_t>(identity_id) * kDim;
  struct Item {
    uint32_t row;
    float s;
    uint64_t t;
  };
  std::vector<Item> items;
  items.reserve(rows.size());
  for (uint32_t r : rows) items.push_back(Item{r, dot512(cn, embs_.at(r).v), faces_.at(r).created_at});
  if (sort == FaceSort::score)
    std::sort(items.begin(), items.end(), [](auto& a, auto& b) { return a.s != b.s ? a.s > b.s : a.row < b.row; });
  else
    std::sort(items.begin(), items.end(), [](auto& a, auto& b) { return a.t != b.t ? a.t > b.t : a.row < b.row; });
  for (uint64_t i = offset; i < items.size() && out.size() < limit; ++i) {
    out.push_back(face_from_slot(faces_.at(items[static_cast<size_t>(i)].row)));
    if (scores) scores->push_back(items[static_cast<size_t>(i)].s);
  }
  return out;
}

std::vector<CooccurrenceEntry> Gallery::cooccurring(int64_t identity_id, uint64_t limit) const {
  std::shared_lock lock(mu_);
  std::vector<CooccurrenceEntry> out;
  if (!identity_live(identity_id)) return out;
  std::vector<std::pair<int64_t, uint64_t>> pairs;  // (other identity, image)
  for (uint32_t r : members_[static_cast<size_t>(identity_id)]) {
    const uint64_t img = faces_.at(r).image_id;
    if (img == 0 || img > image_faces_.size()) continue;
    for (uint32_t o : image_faces_[static_cast<size_t>(img - 1)]) {
      const int64_t other = identity_of(faces_.at(o));
      if (other < 0 || other == identity_id) continue;
      pairs.emplace_back(other, img);
    }
  }
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  for (size_t i = 0; i < pairs.size();) {
    size_t j = i;
    while (j < pairs.size() && pairs[j].first == pairs[i].first) ++j;
    out.push_back(CooccurrenceEntry{pairs[i].first, static_cast<uint32_t>(j - i)});
    i = j;
  }
  std::sort(out.begin(), out.end(), [](auto& a, auto& b) {
    return a.shared_images != b.shared_images ? a.shared_images > b.shared_images : a.identity_id < b.identity_id;
  });
  if (out.size() > limit) out.resize(static_cast<size_t>(limit));
  return out;
}

std::vector<TimelineBucket> Gallery::timeline(int64_t identity_id, int64_t bucket_ms) const {
  std::shared_lock lock(mu_);
  std::vector<TimelineBucket> out;
  if (!identity_live(identity_id) || bucket_ms <= 0) return out;
  std::map<int64_t, uint32_t> buckets;
  for (uint32_t r : members_[static_cast<size_t>(identity_id)]) {
    const int64_t t = static_cast<int64_t>(faces_.at(r).created_at);
    ++buckets[(t / bucket_ms) * bucket_ms];
  }
  for (auto& [start, n] : buckets) out.push_back(TimelineBucket{start, n});
  return out;
}

bool Gallery::identity_centroid(int64_t identity_id, Embedding& out) const {
  std::shared_lock lock(mu_);
  if (!identity_live(identity_id)) return false;
  std::memcpy(out.data(), centroid_norm_.data() + static_cast<size_t>(identity_id) * kDim, sizeof(float) * kDim);
  return true;
}

uint64_t Gallery::identity_count() const {
  std::shared_lock lock(mu_);
  uint64_t n = 0;
  for (uint64_t id = 0; id < identities_.size(); ++id)
    if (slot_live(identities_.at(id).flags)) ++n;
  return n;
}

uint64_t Gallery::largest_identity() const {
  std::shared_lock lock(mu_);
  uint64_t best = 0;
  for (uint64_t id = 0; id < identities_.size(); ++id) {
    const auto& s = identities_.at(id);
    if (slot_live(s.flags)) best = std::max<uint64_t>(best, s.size);
  }
  return best;
}

// ---------------------------------------------------------------------------
// curation

std::optional<int64_t> Gallery::merge_identities(const std::vector<int64_t>& ids_in) {
  std::unique_lock lock(mu_);
  std::vector<int64_t> ids = ids_in;
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  if (ids.size() < 2) return std::nullopt;
  for (int64_t id : ids)
    if (!identity_live(id)) return std::nullopt;
  const int64_t target = ids.front();
  const int64_t now = unix_ms();
  for (size_t i = 1; i < ids.size(); ++i) {
    const int64_t src = ids[i];
    auto rows = members_[static_cast<size_t>(src)];
    for (uint32_t r : rows) {
      faces_.at(r).identity_ref = static_cast<uint64_t>(target) + 1;
      members_[static_cast<size_t>(target)].push_back(r);
    }
    members_[static_cast<size_t>(src)].clear();
    recompute_identity_locked(src, now);  // tombstones it
  }
  recompute_identity_locked(target, now);
  identities_.at(static_cast<uint64_t>(target)).flags |= kIdentityPinned;
  identities_.sync_header();
  centroids_.sync_header();
  return target;
}

std::optional<int64_t> Gallery::split_identity(int64_t identity_id, const std::vector<int64_t>& face_ids) {
  std::unique_lock lock(mu_);
  if (!identity_live(identity_id) || face_ids.empty()) return std::nullopt;
  auto& rows = members_[static_cast<size_t>(identity_id)];
  std::vector<uint32_t> moving;
  for (int64_t fid : face_ids) {
    if (fid < 0 || static_cast<uint64_t>(fid) >= faces_.size()) return std::nullopt;
    if (identity_of(faces_.at(static_cast<uint64_t>(fid))) != identity_id) return std::nullopt;
    moving.push_back(static_cast<uint32_t>(fid));
  }
  std::sort(moving.begin(), moving.end());
  moving.erase(std::unique(moving.begin(), moving.end()), moving.end());
  if (moving.size() >= rows.size()) return std::nullopt;  // must leave something behind
  const int64_t now = unix_ms();
  const int64_t nid = new_identity_locked(now);
  for (uint32_t r : moving) {
    faces_.at(r).identity_ref = static_cast<uint64_t>(nid) + 1;
    members_[static_cast<size_t>(nid)].push_back(r);
  }
  auto& src_rows = members_[static_cast<size_t>(identity_id)];
  src_rows.erase(std::remove_if(src_rows.begin(), src_rows.end(),
                                [&](uint32_t r) { return std::binary_search(moving.begin(), moving.end(), r); }),
                 src_rows.end());
  recompute_identity_locked(identity_id, now);
  recompute_identity_locked(nid, now);
  identities_.at(static_cast<uint64_t>(identity_id)).flags |= kIdentityPinned;
  identities_.at(static_cast<uint64_t>(nid)).flags |= kIdentityPinned;
  identities_.sync_header();
  centroids_.sync_header();
  return nid;
}

bool Gallery::rename_identity(int64_t identity_id, const std::string& name) {
  std::unique_lock lock(mu_);
  if (!identity_live(identity_id)) return false;
  auto& s = identities_.at(static_cast<uint64_t>(identity_id));
  std::memset(s.name, 0, sizeof(s.name));
  std::memcpy(s.name, name.data(), std::min(name.size(), kIdentityNameBytes - 1));
  s.flags |= kIdentityPinned;
  s.updated_at = static_cast<uint64_t>(unix_ms());
  return true;
}

bool Gallery::dissolve_identity(int64_t identity_id) {
  std::unique_lock lock(mu_);
  if (!identity_live(identity_id)) return false;
  for (uint32_t r : members_[static_cast<size_t>(identity_id)]) faces_.at(r).identity_ref = 0;
  members_[static_cast<size_t>(identity_id)].clear();
  recompute_identity_locked(identity_id, unix_ms());
  identities_.at(static_cast<uint64_t>(identity_id)).flags &= ~kIdentityPinned;
  identities_.sync_header();
  return true;
}

bool Gallery::assign_face(int64_t face_id, int64_t identity_id) {
  std::unique_lock lock(mu_);
  if (face_id < 0 || static_cast<uint64_t>(face_id) >= faces_.size()) return false;
  const FaceSlot& f = faces_.at(static_cast<uint64_t>(face_id));
  if (!face_indexable(f.flags)) return false;
  if (identity_id < 0) {
    unassign_locked(static_cast<uint64_t>(face_id));
    return true;
  }
  if (!identity_live(identity_id)) return false;
  assign_locked(static_cast<uint64_t>(face_id), identity_id, true);
  identities_.at(static_cast<uint64_t>(identity_id)).flags |= kIdentityPinned;
  identities_.sync_header();
  return true;
}

// ---------------------------------------------------------------------------
// batch clustering

void Gallery::mark_repeat_images_locked() {
  std::vector<int64_t> ids;
  for (uint64_t i = 0; i < images_.size(); ++i) {
    ImageSlot& im = images_.at(i);
    if (!slot_live(im.flags)) continue;
    ids.clear();
    for (uint32_t r : image_faces_[static_cast<size_t>(i)]) {
      const int64_t id = identity_of(faces_.at(r));
      if (id >= 0) ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    const bool repeat = std::adjacent_find(ids.begin(), ids.end()) != ids.end();
    im.flags = static_cast<uint16_t>((im.flags & ~kImageRepeatIdentity) | (repeat ? kImageRepeatIdentity : 0));
  }
}

void Gallery::apply_labels_locked(const std::vector<int64_t>& labels, uint64_t n, int64_t now) {
  int64_t max_label = -1;
  for (uint64_t i = 0; i < n; ++i) max_label = std::max(max_label, labels[static_cast<size_t>(i)]);
  while (static_cast<int64_t>(identities_.size()) <= max_label) new_identity_locked(now);
  for (uint64_t i = 0; i < n && i < faces_.size(); ++i) {
    FaceSlot& f = faces_.at(i);
    if (!slot_live(f.flags)) continue;
    const int64_t l = labels[static_cast<size_t>(i)];
    f.identity_ref = l < 0 ? 0 : static_cast<uint64_t>(l) + 1;
  }
  // Rebuild membership from the refs (rows >= n keep whatever they had).
  const uint64_t C = identities_.size();
  members_.assign(static_cast<size_t>(C), {});
  const uint64_t total = std::min(embs_.size(), faces_.size());
  for (uint64_t i = 0; i < total; ++i) {
    const FaceSlot& f = faces_.at(i);
    const int64_t id = identity_of(f);
    if (!face_indexable(f.flags) || id < 0 || static_cast<uint64_t>(id) >= C) continue;
    members_[static_cast<size_t>(id)].push_back(static_cast<uint32_t>(i));
  }
  for (uint64_t id = 0; id < C; ++id) {
    if (members_[static_cast<size_t>(id)].empty() && !slot_live(identities_.at(id).flags)) continue;
    recompute_identity_locked(static_cast<int64_t>(id), now);
  }
  mark_repeat_images_locked();
}

std::optional<ClusterReport> Gallery::recluster(const ClusterParams& params) {
  if (clustering_.exchange(true)) return std::nullopt;
  struct Reset {
    std::atomic<bool>& f;
    ~Reset() { f.store(false); }
  } reset{clustering_};
  std::lock_guard<std::mutex> serial(cluster_mu_);

  ClusterReport rep;
  const auto t_all = std::chrono::steady_clock::now();
  const int threads = params.threads > 0 ? params.threads
                                         : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  const int k = std::max(2, params.neighbors);

  // --- snapshot ---
  uint64_t n = 0;
  std::vector<uint32_t> flags;
  std::vector<int64_t> refs;
  std::vector<uint8_t> pinned_identity;
  {
    std::shared_lock lock(mu_);
    n = std::min(embs_.size(), faces_.size());
    flags = flags_cache_;
    flags.resize(static_cast<size_t>(n));
    refs.resize(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; ++i) refs[static_cast<size_t>(i)] = identity_of(faces_.at(i));
    pinned_identity.assign(static_cast<size_t>(identities_.size()), 0);
    for (uint64_t id = 0; id < identities_.size(); ++id) {
      const auto& s = identities_.at(id);
      if (slot_live(s.flags) && (s.flags & kIdentityPinned)) pinned_identity[static_cast<size_t>(id)] = 1;
    }
  }
  std::vector<uint8_t> eligible(static_cast<size_t>(n), 0);
  for (uint64_t i = 0; i < n; ++i) eligible[static_cast<size_t>(i)] = face_indexable(flags[static_cast<size_t>(i)]) ? 1 : 0;
  rep.faces_eligible = static_cast<uint64_t>(std::count(eligible.begin(), eligible.end(), 1));

  // --- kNN over the HNSW, chunked so ingest proceeds between chunks ---
  const auto t_knn = std::chrono::steady_clock::now();
  KnnGraph g;
  g.n = n;
  g.k = k;
  g.nbr.assign(static_cast<size_t>(n) * static_cast<size_t>(k), UINT32_MAX);
  g.sim.assign(static_cast<size_t>(n) * static_cast<size_t>(k), -2.f);
  {
    constexpr uint64_t kChunk = 32768;
    std::vector<uint32_t> qrows;
    std::vector<float> qvecs;
    std::vector<uint64_t> keys;
    std::vector<float> scores;
    const int ef = std::max(128, 2 * (k + 1));
    for (uint64_t start = 0; start < n; start += kChunk) {
      const uint64_t end = std::min(n, start + kChunk);
      qrows.clear();
      for (uint64_t i = start; i < end; ++i)
        if (eligible[static_cast<size_t>(i)]) qrows.push_back(static_cast<uint32_t>(i));
      if (qrows.empty()) continue;
      qvecs.resize(qrows.size() * kDim);
      {
        std::shared_lock lock(mu_);
        for (size_t q = 0; q < qrows.size(); ++q)
          std::memcpy(qvecs.data() + q * kDim, embs_.at(qrows[q]).v, sizeof(float) * kDim);
        keys.resize(qrows.size() * static_cast<size_t>(k + 1));
        scores.resize(keys.size());
        hnsw_.search_many(qvecs.data(), qrows.size(), k + 1, ef, threads, keys.data(), scores.data());
      }
      for (size_t q = 0; q < qrows.size(); ++q) {
        const uint64_t row = qrows[q];
        size_t out = 0;
        for (int j = 0; j <= k && out < static_cast<size_t>(k); ++j) {
          const uint64_t key = keys[q * static_cast<size_t>(k + 1) + static_cast<size_t>(j)];
          if (key == UINT64_MAX || key == row || key >= n) continue;
          g.nbr[static_cast<size_t>(row) * static_cast<size_t>(k) + out] = static_cast<uint32_t>(key);
          g.sim[static_cast<size_t>(row) * static_cast<size_t>(k) + out] =
              scores[q * static_cast<size_t>(k + 1) + static_cast<size_t>(j)];
          ++out;
        }
      }
    }
  }
  rep.knn_ms = ms_since(t_knn);

  // --- label propagation ---
  const auto t_cw = std::chrono::steady_clock::now();
  CsrGraph csr = symmetrize_knn(g, params.edge, eligible);
  rep.edges = csr.edges();
  g.nbr.clear();
  g.nbr.shrink_to_fit();
  g.sim.clear();
  g.sim.shrink_to_fit();
  std::vector<uint32_t> labels(static_cast<size_t>(n));
  std::vector<uint8_t> fixed(static_cast<size_t>(n), 0);
  for (uint64_t i = 0; i < n; ++i) {
    labels[static_cast<size_t>(i)] = static_cast<uint32_t>(i);
    const int64_t ref = refs[static_cast<size_t>(i)];
    if (!eligible[static_cast<size_t>(i)]) {
      fixed[static_cast<size_t>(i)] = 1;
    } else if (ref >= 0 && static_cast<uint64_t>(ref) < pinned_identity.size() && pinned_identity[static_cast<size_t>(ref)]) {
      labels[static_cast<size_t>(i)] = static_cast<uint32_t>(n + static_cast<uint64_t>(ref));
      fixed[static_cast<size_t>(i)] = 1;
    }
  }
  rep.iterations = chinese_whispers(csr, labels, fixed, params.max_iters, threads);
  csr = CsrGraph{};
  // Only eligible nodes get cluster ids; others are unassigned.
  std::vector<uint32_t> elig_labels;
  std::vector<uint32_t> elig_rows;
  elig_labels.reserve(rep.faces_eligible);
  elig_rows.reserve(rep.faces_eligible);
  for (uint64_t i = 0; i < n; ++i) {
    if (!eligible[static_cast<size_t>(i)]) continue;
    elig_labels.push_back(labels[static_cast<size_t>(i)]);
    elig_rows.push_back(static_cast<uint32_t>(i));
  }
  labels.clear();
  labels.shrink_to_fit();
  const uint32_t C = compact_labels(elig_labels);
  rep.clusters_before_merge = C;
  rep.cw_ms = ms_since(t_cw);

  // --- centroid merge ---
  const auto t_merge = std::chrono::steady_clock::now();
  std::vector<float> cent(static_cast<size_t>(C) * kDim, 0.f);
  std::vector<uint32_t> csize(C, 0);
  std::vector<int64_t> cpinned(C, -1);  // pinned identity carried by the cluster
  {
    std::shared_lock lock(mu_);
    for (size_t q = 0; q < elig_rows.size(); ++q) {
      const uint32_t c = elig_labels[q];
      const float* x = embs_.at(elig_rows[q]).v;
      float* dst = cent.data() + static_cast<size_t>(c) * kDim;
      for (int d = 0; d < kDim; ++d) dst[d] += x[d];
      ++csize[c];
      const int64_t ref = refs[elig_rows[q]];
      if (ref >= 0 && static_cast<uint64_t>(ref) < pinned_identity.size() && pinned_identity[static_cast<size_t>(ref)])
        cpinned[c] = ref;
    }
  }
  for (uint32_t c = 0; c < C; ++c) l2_normalize(cent.data() + static_cast<size_t>(c) * kDim);
  UnionFind uf(C);
  std::vector<int64_t> root_pinned = cpinned;
  // Only clusters that can become identities take part in the merge. A
  // singleton has no kNN edge >= edge to anyone, so its centroid is never
  // within `merge` of another cluster's; leaving the (many) singletons out
  // keeps the temporary index small.
  std::vector<uint32_t> cand;
  for (uint32_t c = 0; c < C; ++c)
    if (csize[c] >= params.min_size || cpinned[c] >= 0) cand.push_back(c);
  if (cand.size() >= 2) {
    std::vector<float> ccent(cand.size() * kDim);
    for (size_t i = 0; i < cand.size(); ++i)
      std::memcpy(ccent.data() + i * kDim, cent.data() + static_cast<size_t>(cand[i]) * kDim, sizeof(float) * kDim);
    HnswIndex tmp;
    tmp.rebuild_from(ccent.data(), cand.size(), nullptr, 0);
    const int kk = std::min<int>(11, static_cast<int>(cand.size()));
    std::vector<uint64_t> keys(cand.size() * static_cast<size_t>(kk));
    std::vector<float> scores(keys.size());
    tmp.search_many(ccent.data(), cand.size(), kk, 128, threads, keys.data(), scores.data());
    struct Pair {
      uint32_t a, b;
      float s;
    };
    std::vector<Pair> pairs;
    for (size_t i = 0; i < cand.size(); ++i) {
      const uint32_t c = cand[i];
      for (int j = 0; j < kk; ++j) {
        const uint64_t o = keys[i * static_cast<size_t>(kk) + static_cast<size_t>(j)];
        const float s = scores[i * static_cast<size_t>(kk) + static_cast<size_t>(j)];
        if (o == UINT64_MAX || o >= cand.size() || cand[o] == c || s < params.merge) continue;
        const uint32_t oc = cand[o];
        pairs.push_back(Pair{std::min(c, oc), std::max(c, oc), s});
      }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& x, const Pair& y) { return x.s > y.s; });
    for (const auto& p : pairs) {
      const uint32_t ra = uf.find(p.a), rb = uf.find(p.b);
      if (ra == rb) continue;
      // two hand-curated identities never merge automatically
      if (root_pinned[ra] >= 0 && root_pinned[rb] >= 0 && root_pinned[ra] != root_pinned[rb]) continue;
      uf.unite(ra, rb);
      const uint32_t r = uf.find(ra);
      root_pinned[r] = std::max(root_pinned[ra], root_pinned[rb]);
      ++rep.merged;
    }
  }
  // final cluster id per eligible row
  std::vector<uint32_t> final_of_cluster(C);
  for (uint32_t c = 0; c < C; ++c) final_of_cluster[c] = uf.find(c);
  std::vector<uint32_t> final_labels(elig_labels.size());
  for (size_t q = 0; q < elig_labels.size(); ++q) final_labels[q] = final_of_cluster[elig_labels[q]];
  const uint32_t F = compact_labels(final_labels);
  rep.merge_ms = ms_since(t_merge);

  // --- stable identity ids ---
  const auto t_apply = std::chrono::steady_clock::now();
  std::vector<std::vector<uint32_t>> fmembers(F);
  for (size_t q = 0; q < final_labels.size(); ++q) fmembers[final_labels[q]].push_back(elig_rows[q]);
  std::vector<int64_t> fpinned(F, -1);
  std::vector<std::unordered_map<int64_t, uint32_t>> overlap(F);
  for (uint32_t f = 0; f < F; ++f) {
    for (uint32_t r : fmembers[f]) {
      const int64_t ref = refs[r];
      if (ref < 0) continue;
      if (static_cast<uint64_t>(ref) < pinned_identity.size() && pinned_identity[static_cast<size_t>(ref)]) fpinned[f] = ref;
      ++overlap[f][ref];
    }
  }
  std::vector<uint32_t> order(F);
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    return fmembers[a].size() != fmembers[b].size() ? fmembers[a].size() > fmembers[b].size() : a < b;
  });
  std::vector<int64_t> fid(F, -1);
  std::unordered_set<int64_t> claimed;
  for (uint32_t f : order)
    if (fpinned[f] >= 0) {
      fid[f] = fpinned[f];
      claimed.insert(fpinned[f]);
    }
  int64_t next_id = 0;
  {
    std::shared_lock lock(mu_);
    next_id = static_cast<int64_t>(identities_.size());
  }
  for (uint32_t f : order) {
    if (fid[f] >= 0) continue;
    int64_t best = -1;
    uint32_t best_n = 0;
    for (const auto& [id, cnt] : overlap[f]) {
      if (claimed.count(id)) continue;
      if (cnt > best_n || (cnt == best_n && id < best)) {
        best = id;
        best_n = cnt;
      }
    }
    if (best >= 0) {
      fid[f] = best;
      claimed.insert(best);
    } else {
      fid[f] = next_id++;
    }
  }
  std::vector<int64_t> row_labels(static_cast<size_t>(n), -1);
  for (uint32_t f = 0; f < F; ++f) {
    if (fpinned[f] < 0 && fmembers[f].size() < params.min_size) {
      ++rep.singletons_dropped;
      continue;
    }
    for (uint32_t r : fmembers[f]) row_labels[r] = fid[f];
  }

  {
    std::unique_lock lock(mu_);
    apply_labels_locked(row_labels, n, unix_ms());
    faces_.sync_header();
    identities_.sync_header();
    centroids_.sync_header();
    images_.sync_header();
    rep.identities = 0;
    rep.identities_pinned = 0;
    rep.largest = 0;
    rep.faces_assigned = 0;
    for (uint64_t id = 0; id < identities_.size(); ++id) {
      const auto& s = identities_.at(id);
      if (!slot_live(s.flags)) continue;
      ++rep.identities;
      if (s.flags & kIdentityPinned) ++rep.identities_pinned;
      rep.largest = std::max<uint64_t>(rep.largest, s.size);
      rep.faces_assigned += s.size;
    }
  }
  rep.apply_ms = ms_since(t_apply);
  rep.total_ms = ms_since(t_all);
  spdlog::info(
      "recluster eligible={} edges={} clusters={} merged={} small_dropped={} identities={} assigned={} largest={} "
      "iters={} knn={:.0f}ms cw={:.0f}ms merge={:.0f}ms apply={:.0f}ms total={:.0f}ms",
      rep.faces_eligible, rep.edges, rep.clusters_before_merge, rep.merged, rep.singletons_dropped, rep.identities,
      rep.faces_assigned, rep.largest, rep.iterations, rep.knn_ms, rep.cw_ms, rep.merge_ms, rep.apply_ms,
      rep.total_ms);
  return rep;
}

// ---------------------------------------------------------------------------
// impostor evaluation

ImpostorReport Gallery::impostor_eval(uint64_t max_pairs, uint64_t seed) const {
  std::shared_lock lock(mu_);
  ImpostorReport r;
  std::mt19937_64 rng(seed);
  std::vector<uint64_t> imgs;
  for (uint64_t i = 0; i < images_.size(); ++i) {
    if (!slot_live(images_.at(i).flags)) continue;
    uint32_t good = 0;
    for (uint32_t row : image_faces_[static_cast<size_t>(i)])
      if (face_indexable(faces_.at(row).flags)) ++good;
    if (good >= 2) imgs.push_back(i);
  }
  std::shuffle(imgs.begin(), imgs.end(), rng);
  std::vector<float> same;
  std::vector<uint32_t> rows;
  for (uint64_t i : imgs) {
    if (same.size() >= max_pairs) break;
    rows.clear();
    for (uint32_t row : image_faces_[static_cast<size_t>(i)])
      if (face_indexable(faces_.at(row).flags)) rows.push_back(row);
    if (rows.size() > 8) {
      std::shuffle(rows.begin(), rows.end(), rng);
      rows.resize(8);
    }
    bool used = false;
    for (size_t a = 0; a < rows.size(); ++a)
      for (size_t b = a + 1; b < rows.size(); ++b) {
        if (iou(faces_.at(rows[a]), faces_.at(rows[b])) > 0.05f) continue;
        same.push_back(dot512(embs_.at(rows[a]).v, embs_.at(rows[b]).v));
        used = true;
      }
    if (used) ++r.images_used;
  }
  r.same_image_pairs = same.size();

  std::vector<uint32_t> pool;
  for (uint64_t i = 0; i < std::min(embs_.size(), faces_.size()); ++i)
    if (face_indexable(faces_.at(i).flags)) pool.push_back(static_cast<uint32_t>(i));
  std::vector<float> rnd;
  if (pool.size() >= 2) {
    std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
    const uint64_t want = std::max<uint64_t>(same.size(), std::min<uint64_t>(max_pairs, 200000));
    while (rnd.size() < want) {
      const uint32_t a = pool[pick(rng)], b = pool[pick(rng)];
      if (a == b || faces_.at(a).image_id == faces_.at(b).image_id) continue;
      rnd.push_back(dot512(embs_.at(a).v, embs_.at(b).v));
    }
  }
  r.random_pairs = rnd.size();

  auto mean_sd = [](const std::vector<float>& v, double& m, double& sd) {
    m = sd = 0;
    if (v.empty()) return;
    for (float x : v) m += x;
    m /= static_cast<double>(v.size());
    for (float x : v) sd += (x - m) * (x - m);
    sd = std::sqrt(sd / static_cast<double>(std::max<size_t>(1, v.size() - 1)));
  };
  mean_sd(same, r.same_image_mean, r.same_image_sd);
  mean_sd(rnd, r.random_mean, r.random_sd);
  if (!rnd.empty()) {
    constexpr int bins = 300;
    std::vector<uint32_t> hist(bins, 0);
    for (float x : rnd) {
      int b = static_cast<int>((x + 0.5f) / 1.5f * bins);
      b = std::clamp(b, 0, bins - 1);
      ++hist[static_cast<size_t>(b)];
    }
    const int mb = static_cast<int>(std::max_element(hist.begin(), hist.end()) - hist.begin());
    r.random_mode = -0.5 + (mb + 0.5) * 1.5 / bins;
  }
  const float thresholds[] = {0.20f, 0.25f, 0.30f, 0.35f, 0.40f, 0.45f, 0.50f, 0.55f, 0.60f, 0.65f, 0.70f, 0.80f, 0.90f};
  r.impostor_ceiling = 1.f;
  bool ceiling_set = false;
  for (float t : thresholds) {
    double far_same = 0, above = 0, mirror = 0;
    if (!same.empty()) far_same = static_cast<double>(std::count_if(same.begin(), same.end(), [&](float x) { return x > t; })) / static_cast<double>(same.size());
    if (!rnd.empty()) {
      above = static_cast<double>(std::count_if(rnd.begin(), rnd.end(), [&](float x) { return x > t; })) / static_cast<double>(rnd.size());
      const double left = 2.0 * r.random_mode - t;
      mirror = static_cast<double>(std::count_if(rnd.begin(), rnd.end(), [&](float x) { return x < left; })) / static_cast<double>(rnd.size());
    }
    r.same_image_far.emplace_back(t, far_same);
    r.random_above.emplace_back(t, above);
    r.random_excess.emplace_back(t, std::max(0.0, above - mirror));
    if (!ceiling_set && !rnd.empty() && mirror == 0.0) {
      r.impostor_ceiling = t;
      ceiling_set = true;
    }
  }
  return r;
}

}  // namespace hvax
