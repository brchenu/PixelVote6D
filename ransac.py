import torch

class PVNetRansac:
    def __init__(self, mask: torch.Tensor, vfield: torch.Tensor, num_iter: int):
        # vfield shape is (B*2, H, W)
        self.num_keypoints = vfield.size(0) // 2
        self.batch_size = self.num_keypoints

        x_components = vfield[: self.num_keypoints]
        y_components = vfield[self.num_keypoints :]
        self.vfield = torch.stack([x_components, y_components], dim=1)  # B, 2, H, W

        self.num_iter = num_iter
        self.valid_index = mask.nonzero()  # N, 2

        print(f"shape valid idx: {self.valid_index.shape}")

    def batched_hypothesis(self):
        """Batch hypothesis over B keypoits"""

        assert self.vfield.dim() == 4 and self.vfield.size(1) == 2  # B, 2, H, W

        idx = torch.randperm(self.valid_index.size(0))[:2]  # 2
        p1 = self.valid_index[idx[0]]
        p2 = self.valid_index[idx[1]]

        v1 = self.vfield[:, :, p1[0], p1[1]]
        v2 = self.vfield[:, :, p2[0], p2[1]]

        assert v1.dim() == 2
        assert v1.size(0) == self.batch_size
        assert v1.size(1) == 2

        # convert row, col to x, y
        p1 = p1.flip(0).float()
        p2 = p2.flip(0).float()

        assert p1.dim() == 1 and p1.size(0) == 2

        # Vectorized intersect line

        A = torch.stack([v1, -v2], dim=-1)  # B, 2, 2
        print(f"shape A: {A.shape}")
        b = (p2 - p1).unsqueeze(0).repeat(self.batch_size, 1).float()  # B, 2
        print(f"shape b: {b.shape}")

        assert A.dim() == 3 and A.size(1) == 2 and A.size(2) == 2
        assert b.dim() == 2 and b.size(1) == 2

        t = torch.linalg.solve(A, b)
        print(f"shape t: {t.shape}")

        assert t.size(0) == self.batch_size

        return p1 + t[:, 0:1] * v1

    def score(self, hypothesis: torch.Tensor):
        assert hypothesis.dim() == 2
        assert hypothesis.size(1) == 2  # B, 2

        h, w = self.vfield.size()[2:]
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        coords = torch.stack([x, y], dim=0).float()  # 2, H, W

        hypothesis = hypothesis[:, :, None, None]  # B, 2, 1, 1
        coords = coords[None, :, :, :].repeat(hypothesis.size(0), 1, 1, 1)  # B, 2, H, W

        hypo_vec = hypothesis - coords  # B, 2, H, W
        hypo_vec = hypo_vec / torch.norm(hypo_vec, dim=1, keepdim=True)  # B, 2, H, W

        # hypo_vec = hypo_vec.reshape(hypo_vec.size(0), 2, -1)
        data_vec = self.vfield.reshape(self.batch_size, 2, -1)

        assert hypo_vec.dim() == 3 and hypo_vec.size(1) == 2
        assert data_vec.dim() == 3 and data_vec.size(1) == 2

        scores = torch.sum(hypo_vec * data_vec, dim=1)  # B, H*W
        # scores = hypo_vec @ data_vec  # B, H*W

        THRESHOLD = 0.9
        inliers = scores > THRESHOLD

        consensus = inliers.sum(dim=1)  # B

        return consensus, hypothesis.squeeze()  # B, 2

    def ransac(self):

        all_hyopt = []
        all_consensus = []
        for _ in range(self.num_iter):
            hypo_points = self.batched_hypothesis()
            consensus, hypo_points = self.score(hypo_points)
            all_hyopt.append(hypo_points)
            all_consensus.append(consensus)

        all_hyopt = torch.stack(all_hyopt, dim=0)  # I, B, 2
        all_consensus = torch.stack(all_consensus, dim=0)  # I, B

        # get max of each batch for iter
        best_consensus, best_idx = all_consensus.max(dim=0)  # B
        best_hypo_points = all_hyopt[best_idx, :, :]  # B, 2