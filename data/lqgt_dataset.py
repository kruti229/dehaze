import os, cv2, numpy as np, torch
from torch.utils.data import Dataset
 
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
 
class LQGTDataset(Dataset):
    def __init__(self, opt, split="train"):
        self.opt = opt
        self.patch = opt['datasets'][split].get('patch_size', None)
        self.lq_paths = self._scan(opt['datasets'][split]['dataroot_LQ'])
        self.gt_paths = self._scan(opt['datasets'][split]['dataroot_GT'])
        assert len(self.lq_paths) == len(self.gt_paths), "LQ/GT count mismatch"
        print(f"[Dataset] {split}: {len(self.lq_paths)} pairs")
 
    def _scan(self, root):
        files = [os.path.join(root, f) for f in sorted(os.listdir(root))
                 if os.path.splitext(f)[1].lower() in IMG_EXTS]
        return files
 
    def __len__(self): return len(self.lq_paths)
 
    def __getitem__(self, idx):
        lq = self._load(self.lq_paths[idx])
        gt = self._load(self.gt_paths[idx])
        if self.patch:
            lq, gt = self._crop(lq, gt, self.patch)
        lq, gt = self._augment(lq, gt)
        return {
            "LQ": self._to_tensor(lq),
            "GT": self._to_tensor(gt),
            "LQ_path": self.lq_paths[idx]
        }
 
    def _load(self, path):
        img = cv2.imread(path)
        if img is None: raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    def _crop(self, lq, gt, size):
        H, W = lq.shape[:2]
        top  = np.random.randint(0, max(1, H - size))
        left = np.random.randint(0, max(1, W - size))
        return (lq[top:top+size, left:left+size],
                gt[top:top+size, left:left+size])
 
    def _augment(self, lq, gt):
        if np.random.rand() > 0.5:  # horizontal flip
            lq, gt = lq[:, ::-1], gt[:, ::-1]
        return lq.copy(), gt.copy()
 
    def _to_tensor(self, img):
        t = torch.from_numpy(img.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1)  # [3, H, W]
