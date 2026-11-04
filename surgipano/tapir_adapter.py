"""Persistent causal BootsTAPIR cohorts with stable identities and adaptive births.

Each cohort has independent causal context. Existing cohorts are never reset on a
fixed timer; only fully stale cohorts are retired. Newly born queries are added
without altering the feature/state tensors of old cohorts.
"""
from dataclasses import dataclass
import cv2
import numpy as np
import torch
import tree
from tapnet.torch import tapir_model


@dataclass
class TrackBatch:
    prev: np.ndarray
    curr: np.ndarray
    confidence: np.ndarray
    ids: np.ndarray
    age: np.ndarray
    born: int = 0
    active: int = 0
    occluded: int = 0
    retired: int = 0

    def __iter__(self):
        # Backward-compatible unpacking (prev, curr, count).
        yield self.prev
        yield self.curr
        yield len(self.curr)


class TapirTracker:
    def __init__(self, checkpoint, device='cuda', resolution=256, spacing=48,
                 refresh=None, max_points=192, min_confidence=0.5,
                 max_missing=90, replenish_every=5, min_distance=None):
        self.device = torch.device(device)
        self.resolution = int(resolution)
        self.spacing = int(spacing)
        self.max_points = int(max_points)
        self.min_confidence = float(min_confidence)
        self.max_missing = int(max_missing)
        self.replenish_every = max(1, int(replenish_every))
        self.min_distance = float(min_distance or spacing * .7)
        self.model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=False))
        self.model.eval()
        self.cohorts = []
        self.next_id = 0
        self.step = 0
        self.last = None
        self.canvas_anchors = {}  # ID -> panorama position at most recent reliable observation

    def _frame(self, rgb):
        x = cv2.resize(rgb, (self.resolution, self.resolution)).astype(np.float32)
        return torch.from_numpy(x).to(self.device)[None, None] / 255 * 2 - 1

    def _candidate_points(self, mask, occupied, budget):
        h, w = mask.shape
        if budget <= 0:
            return np.empty((0, 2), np.float32)
        # Offset grids across births to cover holes without clustering.
        offset = (self.step // self.replenish_every) % 2
        shift = self.spacing // 4 if offset else self.spacing // 2
        ys, xs = np.mgrid[shift:h:self.spacing, shift:w:self.spacing]
        candidates = np.stack((xs.ravel(), ys.ravel()), axis=1).astype(np.float32)
        candidates = candidates[mask[candidates[:, 1].astype(int), candidates[:, 0].astype(int)] > 0]
        selected = []
        used = np.asarray(occupied, np.float32).reshape(-1, 2)
        # Choose sparsest candidate first; newly selected points also repel births.
        while len(candidates) and len(selected) < budget:
            if len(used):
                d2 = ((candidates[:, None] - used[None]) ** 2).sum(axis=2).min(axis=1)
                k = int(np.argmax(d2))
                if d2[k] < self.min_distance ** 2:
                    break
            else:
                k = len(candidates) // 2
            point = candidates[k]
            selected.append(point)
            used = np.vstack([used, point.reshape(1, 2)])
            candidates = np.delete(candidates, k, axis=0)
        return np.asarray(selected, np.float32).reshape(-1, 2)

    @torch.inference_mode()
    def _birth(self, frame, grids, points, shape):
        if not len(points):
            return 0
        h, w = shape
        query = np.stack([np.zeros(len(points)), points[:, 1] * self.resolution / h,
                          points[:, 0] * self.resolution / w], axis=1).astype(np.float32)
        features = self.model.get_query_features(frame, is_training=False,
                         query_points=torch.from_numpy(query)[None].to(self.device), feature_grids=grids)
        state = self.model.construct_initial_causal_state(len(points), len(features.resolutions) - 1)
        state = tree.map_structure(lambda x: x.to(self.device), state)
        ids = np.arange(self.next_id, self.next_id + len(points), dtype=np.int64)
        self.next_id += len(points)
        self.cohorts.append(dict(features=features, state=state, ids=ids, prev=points,
                                 confidence=np.ones(len(points), np.float32),
                                 missing=np.zeros(len(points), np.int32),
                                 birth=np.full(len(points), self.step, np.int32),
                                 alive=np.ones(len(points), bool), valid=np.ones(len(points), bool)))
        return len(points)

    @torch.inference_mode()
    def update(self, rgb, mask):
        h, w = rgb.shape[:2]
        frame = self._frame(rgb)
        grids = self.model.get_feature_grids(frame, is_training=False)
        prev_list, curr_list, conf_list, id_list, age_list = [], [], [], [], []
        retired = 0
        occupied = []
        for cohort in self.cohorts:
            result = self.model.estimate_trajectories(frame.shape[-3:-1], is_training=False,
                feature_grids=grids, query_features=cohort['features'], query_points_in_video=None,
                query_chunk_size=64, causal_context=cohort['state'], get_causal_context=True)
            cohort['state'] = result['causal_context']
            xy = result['tracks'][-1].detach().cpu().numpy()[0, :, -1, :].copy()
            xy[:, 0] *= w / self.resolution
            xy[:, 1] *= h / self.resolution
            occ = result['occlusion'][-1][0, :, -1].detach().cpu().numpy()
            dist = result['expected_dist'][-1][0, :, -1].detach().cpu().numpy()
            sigmoid = lambda a: 1. / (1. + np.exp(-np.clip(a, -30, 30)))
            conf = ((1 - sigmoid(occ)) * (1 - sigmoid(dist))).astype(np.float32)
            finite = np.isfinite(xy).all(axis=1)
            inside = finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            masked = np.zeros(len(xy), bool)
            idx = np.where(inside)[0]
            masked[idx] = mask[xy[idx, 1].astype(int), xy[idx, 0].astype(int)] > 0
            good = cohort['alive'] & inside & masked & (conf >= self.min_confidence)
            pairs = good & cohort['valid']
            if pairs.any():
                prev_list.append(cohort['prev'][pairs].copy())
                curr_list.append(xy[pairs].copy())
                conf_list.append(conf[pairs].copy())
                id_list.append(cohort['ids'][pairs].copy())
                age_list.append((self.step - cohort['birth'][pairs]).copy())
            cohort['missing'][good] = 0
            cohort['missing'][~good & cohort['alive']] += 1
            stale = cohort['alive'] & (cohort['missing'] > self.max_missing)
            retired += int(stale.sum())
            cohort['alive'][stale] = False
            # Retain the last valid position when occluded. Never use an invalid
            # previous coordinate for next-frame geometric correspondences.
            cohort['prev'][good] = xy[good]
            cohort['valid'] = good
            cohort['confidence'] = conf
            if good.any():
                occupied.extend(xy[good].tolist())
        self.cohorts = [c for c in self.cohorts if c['alive'].any()]
        active = sum(int(c['alive'].sum()) for c in self.cohorts)
        born = 0
        if self.step == 0 or self.step % self.replenish_every == 0 or len(occupied) < max(4, self.max_points // 3):
            # Stale/occluded points reserve their identities but not their visible
            # spatial cells; cap the total memory footprint including occlusions.
            budget = max(0, self.max_points - active)
            points = self._candidate_points(mask, occupied, budget)
            born = self._birth(frame, grids, points, (h, w))
        empty2 = np.empty((0, 2), np.float32)
        cat = lambda seq, empty: np.concatenate(seq) if seq else empty
        batch = TrackBatch(cat(prev_list, empty2), cat(curr_list, empty2),
                           cat(conf_list, np.empty(0, np.float32)),
                           cat(id_list, np.empty(0, np.int64)),
                           cat(age_list, np.empty(0, np.int32)), born=born,
                           active=active + born, occluded=active - len(occupied), retired=retired)
        self.last = batch
        self.step += 1
        return batch

    def remember_canvas(self, H, batch=None):
        """Record persistent ID -> panorama observation without resetting identities."""
        from common import transform
        batch = self.last if batch is None else batch
        if H is None or batch is None or not len(batch.curr):
            return
        xy = transform(H, batch.curr)
        for ident, point in zip(batch.ids, xy):
            if np.isfinite(point).all():
                self.canvas_anchors[int(ident)] = point.copy()
