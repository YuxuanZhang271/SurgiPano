"""EMDQ-inspired robust 2D residual deformation field (not full dual-quaternion EMDQ).

Coordinates: current-frame p -> previous-frame H(p)+D(p). Local field uses
iteratively reweighted Gaussian kernel regression and a confidence taper.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from common import transform


@dataclass
class ResidualField:
    nodes: np.ndarray
    displacement: np.ndarray
    weights: np.ndarray
    sigma: float
    max_displacement: float
    support: float

    def sample(self, xy):
        xy = np.asarray(xy, np.float32).reshape(-1, 2)
        if not len(self.nodes):
            return np.zeros_like(xy)
        d2 = ((xy[:, None] - self.nodes[None]) ** 2).sum(axis=2)
        kernel = np.exp(-d2 / (2 * self.sigma ** 2)) * self.weights[None]
        denom = kernel.sum(axis=1)
        # A distant node should not induce deformation everywhere.
        reliability = np.clip(denom / (denom + self.support), 0, 1)
        disp = kernel @ self.displacement / np.maximum(denom[:, None], 1e-8)
        disp *= reliability[:, None]
        length = np.linalg.norm(disp, axis=1)
        disp *= np.minimum(1., self.max_displacement / np.maximum(length, 1e-6))[:, None]
        return disp.astype(np.float32)


def fit_residual_field(H, prev, curr, confidence, sigma=65., max_residual=20.,
                       iterations=5, min_points=6, support=.15):
    """Robustly fit local residuals after global alignment.

    Residuals that are grossly inconsistent with their local neighbors are
    downweighted. Insufficient spatial evidence returns no local correction.
    """
    if H is None or len(curr) < min_points:
        return None, 0, float('nan')
    curr = np.asarray(curr, np.float32)
    prev = np.asarray(prev, np.float32)
    residual = prev - transform(H, curr)
    norm = np.linalg.norm(residual, axis=1)
    finite = np.isfinite(residual).all(axis=1)
    valid = finite & (norm < max_residual * 3)
    if valid.sum() < min_points:
        return None, 0, float('nan')
    xy, r = curr[valid], residual[valid]
    weights = np.clip(np.asarray(confidence, np.float32)[valid], .01, 1.)
    d2 = ((xy[:, None] - xy[None]) ** 2).sum(axis=2)
    spatial = np.exp(-d2 / (2 * sigma ** 2))
    np.fill_diagonal(spatial, 0.)  # leave-one-out: do not self-confirm outliers
    for _ in range(iterations):
        local = spatial * weights[None]
        denom = local.sum(axis=1)
        prediction = local @ r / np.maximum(denom[:, None], 1e-8)
        error = np.linalg.norm(r - prediction, axis=1)
        scale = max(1.5, float(np.median(error)) * 1.4826)
        robust = 1. / (1. + (error / (2.5 * scale)) ** 2)
        robust[denom < .05] *= .25
        weights = np.clip(np.asarray(confidence, np.float32)[valid], .01, 1.) * robust
    good = weights > .15
    if good.sum() < min_points:
        return None, 0, float('nan')
    field = ResidualField(xy[good], r[good], weights[good], float(sigma),
                          float(max_residual), float(support))
    fit = field.sample(xy[good])
    rmse = float(np.sqrt(np.mean(np.sum((fit - r[good]) ** 2, axis=1))))
    # A large residual prediction error indicates non-coherent tissue motion
    # or mismatches; do not deform the panorama in that case.
    if not np.isfinite(rmse) or rmse > min(5., max_residual * .25):
        return None, int(good.sum()), rmse
    return field, int(good.sum()), rmse


def compose_nonrigid(canvas, coverage, frame, H, field, mask=None,
                      alpha=.9, grid_step=8):
    """Inverse-map a canvas ROI with H^-1 and fixed-point local correction.

    Inverse mapping is evaluated only in the projected frame bounding box;
    this avoids allocating a full-frame-size float map for large panoramas.
    """
    if field is None:
        from common import compose
        return compose(canvas, coverage, frame, H, mask, alpha)
    h, w = frame.shape[:2]
    if mask is None:
        mask = np.full((h, w), 255, np.uint8)
    corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    projected = transform(H, corners)
    margin = int(field.max_displacement * 3 + 5)
    x0 = max(0, int(np.floor(projected[:, 0].min())) - margin)
    y0 = max(0, int(np.floor(projected[:, 1].min())) - margin)
    x1 = min(canvas.shape[1], int(np.ceil(projected[:, 0].max())) + margin + 1)
    y1 = min(canvas.shape[0], int(np.ceil(projected[:, 1].max())) + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return 0
    inv = np.linalg.inv(H)
    # Coarse inverse mesh: evaluate residual only at mesh nodes, then resize.
    step = max(1, int(grid_step))
    xs = np.unique(np.r_[np.arange(x0, x1, step), x1-1]).astype(np.float32)
    ys = np.unique(np.r_[np.arange(y0, y1, step), y1-1]).astype(np.float32)
    gx, gy = np.meshgrid(xs, ys)
    targets = np.stack([gx.ravel(), gy.ravel()], axis=1)
    source = transform(inv, targets)
    for _ in range(4):
        correction = field.sample(source)
        source = transform(inv, targets - correction)
    # Interpolate in true mesh coordinates (including the short boundary cell).
    fullx, fully = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    px = np.interp(fullx.ravel(), xs, np.arange(len(xs))).reshape(fullx.shape).astype(np.float32)
    py = np.interp(fully.ravel(), ys, np.arange(len(ys))).reshape(fully.shape).astype(np.float32)
    mapx = cv2.remap(source[:, 0].reshape(len(ys), len(xs)), px, py, cv2.INTER_LINEAR)
    mapy = cv2.remap(source[:, 1].reshape(len(ys), len(xs)), px, py, cv2.INTER_LINEAR)
    warped = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.remap(mask, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    region = canvas[y0:y1, x0:x1]
    cov = coverage[y0:y1, x0:x1]
    fresh = valid & (cov == 0)
    overlap = valid & (cov != 0)
    region[fresh] = warped[fresh]
    region[overlap] = np.clip(alpha * region[overlap].astype(np.float32) +
                              (1-alpha) * warped[overlap].astype(np.float32), 0, 255).astype(np.uint8)
    cov[valid] = 255
    return int(valid.sum())
