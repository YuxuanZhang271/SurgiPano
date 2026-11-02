"""Online causal BootsTAPIR wrapper based on uploaded pytorch_video.py."""
import numpy as np
import cv2
import torch
import tree
from tapnet.torch import tapir_model

class TapirTracker:
    def __init__(self, checkpoint, device='cuda', resolution=256, spacing=48, refresh=30):
        self.device = torch.device(device)
        self.resolution = resolution
        self.spacing = spacing
        self.refresh = refresh
        self.model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=False))
        self.model.eval()
        self.features = self.state = self.prev = None
        self.age = 0

    def _frame(self, rgb):
        x = cv2.resize(rgb, (self.resolution, self.resolution)).astype(np.float32)
        return torch.from_numpy(x).to(self.device)[None, None] / 255 * 2 - 1

    @torch.inference_mode()
    def _initialize(self, rgb, mask):
        h, w = rgb.shape[:2]
        ys, xs = np.mgrid[self.spacing//2:h:self.spacing, self.spacing//2:w:self.spacing]
        points = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
        good = mask[points[:,1].astype(int), points[:,0].astype(int)] > 0
        points = points[good]
        if len(points) < 4: raise RuntimeError('Not enough valid TAPIR query points; reduce --tapir-spacing or mask erosion')
        query = np.stack([np.zeros(len(points)), points[:,1]*self.resolution/h,
                          points[:,0]*self.resolution/w], axis=1).astype(np.float32)
        frame = self._frame(rgb)
        grids = self.model.get_feature_grids(frame, is_training=False)
        self.features = self.model.get_query_features(frame, is_training=False,
                        query_points=torch.from_numpy(query)[None].to(self.device), feature_grids=grids)
        self.state = self.model.construct_initial_causal_state(len(points), len(self.features.resolutions)-1)
        self.state = tree.map_structure(lambda x: x.to(self.device), self.state)
        self.prev = points
        self.age = 0

    @torch.inference_mode()
    def update(self, rgb, mask):
        """Return (prev_xy, current_xy, valid_count), all in cropped frame pixels."""
        if self.features is None or self.age >= self.refresh:
            self._initialize(rgb, mask)
            return np.empty((0,2), np.float32), np.empty((0,2), np.float32), 0
        h, w = rgb.shape[:2]
        frame = self._frame(rgb)
        grids = self.model.get_feature_grids(frame, is_training=False)
        result = self.model.estimate_trajectories(frame.shape[-3:-1], is_training=False,
                   feature_grids=grids, query_features=self.features, query_points_in_video=None,
                   query_chunk_size=64, causal_context=self.state, get_causal_context=True)
        self.state = result['causal_context']
        track = result['tracks'][-1].detach().cpu().numpy()[0, :, -1, :].copy()
        track[:,0] *= w / self.resolution
        track[:,1] *= h / self.resolution
        occ = result['occlusion'][-1][0, :, -1].detach().cpu().numpy()
        dist = result['expected_dist'][-1][0, :, -1].detach().cpu().numpy()
        sigmoid = lambda a: 1/(1+np.exp(-np.clip(a, -30, 30)))
        visible = (1-sigmoid(occ))*(1-sigmoid(dist)) > 0.5
        inside = ((track[:,0] >= 0) & (track[:,0] < w) &
                  (track[:,1] >= 0) & (track[:,1] < h))
        idx = np.where(inside)[0]
        accepted = np.zeros(len(track), bool)
        accepted[idx] = mask[track[idx,1].astype(int), track[idx,0].astype(int)] > 0
        ok = visible & accepted & np.isfinite(track).all(axis=1)
        prev, curr = self.prev[ok].copy(), track[ok].copy()
        # Track query identities; do not discard occluded points from causal state.
        self.prev = track
        self.age += 1
        return prev, curr, len(curr)
