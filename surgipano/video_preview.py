"""Side-by-side original frame and panorama with projected current-frame ROI."""
from pathlib import Path
import cv2
import numpy as np
from common import normalize_h, transform, image_numpy

class AlignmentVideo:
    def __init__(self, path, fps, width=1600, height=720, every=1):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = width, height
        self.every = max(1, int(every))
        self.count = 0
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), max(1., float(fps))/self.every, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f'Cannot open video writer: {path}')

    @staticmethod
    def fit(image, width, height):
        ih, iw = image.shape[:2]
        s = min(width/max(iw,1), height/max(ih,1))
        nw, nh = max(1,round(iw*s)), max(1,round(ih*s))
        return cv2.resize(image,(nw,nh)), nw, nh

    def write(self, original_rgb, canvas_rgb, H, frame_shape, frame_index, label='', coverage=None):
        self.count += 1
        if (self.count-1) % self.every:
            return
        pano = image_numpy(canvas_rgb)
        if pano is None: return
        pano = cv2.cvtColor(pano, cv2.COLOR_RGB2BGR)
        H = normalize_h(H)
        if H is not None:
            fh, fw = frame_shape[:2]
            corners = np.float32([[0,0],[fw-1,0],[fw-1,fh-1],[0,fh-1]])
            try:
                quad = transform(H,corners)
                if np.isfinite(quad).all() and np.max(np.abs(quad)) < 1e7:
                    cv2.polylines(pano,[np.round(quad).astype(np.int32)],True,(0,255,255),3,cv2.LINE_AA)
            except cv2.error: pass
        # Crop panorama to occupied pixels; include current projected ROI when valid.
        if coverage is not None:
            mask = image_numpy(coverage)
            if mask.ndim == 3: mask = mask[:,:,0]
            ys,xs = np.where(mask > 0)
        else:
            ys,xs = np.where(np.any(pano != 0,axis=2))
        if len(xs):
            pad=35
            x0=max(0,int(xs.min())-pad); x1=min(pano.shape[1],int(xs.max())+pad+1)
            y0=max(0,int(ys.min())-pad); y1=min(pano.shape[0],int(ys.max())+pad+1)
            pano=pano[y0:y1,x0:x1]
        original = cv2.cvtColor(original_rgb,cv2.COLOR_RGB2BGR)
        left_w = self.width//2
        right_w = self.width-left_w
        out = np.zeros((self.height,self.width,3),np.uint8)
        for img,xoff,bw in [(original,0,left_w),(pano,left_w,right_w)]:
            fitted,nw,nh=self.fit(img,bw,self.height)
            x=xoff+(bw-nw)//2; y=(self.height-nh)//2
            out[y:y+nh,x:x+nw]=fitted
        cv2.line(out,(left_w,0),(left_w,self.height),(110,110,110),2)
        for title,x in [('Original',12),('Aligned panorama',left_w+12)]:
            cv2.putText(out,title,(x,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA)
        status=f'Frame {frame_index} | {label}'
        cv2.putText(out,status,(12,self.height-18),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
        self.writer.write(out)

    def close(self):
        self.writer.release()
