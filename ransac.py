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

        # Pre-compute used in self.scores
        self.mask_vecs = self.vfield[:, :, self.valid_index[:, 0], self.valid_index[:, 1]]  # (B, 2, N)
        # flip valid index to x, y, transpose to 2, N
        self.mask_coords = self.valid_index.flip(1).float().T  # (2, N)

    def batched_hypothesis(self):
        """Batch hypothesis over B keypoits"""

        assert self.vfield.dim() == 4 and self.vfield.size(1) == 2  # B, 2, H, W
        assert self.valid_index.dim() == 2 
        assert self.valid_index.size(1) == 2  # N, 2

        idx = torch.randperm(self.valid_index.size(0))[:2]
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

        assert p1.dim() == 1
        assert p1.size(0) == 2

        # Vectorized intersect line

        A = torch.stack([v1, -v2], dim=-1)  # B, 2, 2
        b = (p2 - p1).unsqueeze(0).repeat(self.batch_size, 1).float()  # B, 2

        assert A.dim() == 3 and A.size(1) == 2 and A.size(2) == 2
        assert b.dim() == 2 and b.size(1) == 2

        t = torch.linalg.solve(A, b)

        assert t.size(0) == self.batch_size

        return p1 + t[:, 0:1] * v1

    def scores(self, hypothesis: torch.Tensor):

        # vfield (B, 2, H, W)
        # valid_index (N, 2)

        assert hypothesis.size(0) == self.batch_size and hypothesis.size(1) == 2

        dir_to_hipo = hypothesis[:, :, None] - self.mask_coords[None, :, :]  # (B, 2, N)
        dir_to_hipo = torch.nn.functional.normalize(dir_to_hipo, dim=1)  # (B, 2, N)

        dot_products = (dir_to_hipo * self.mask_vecs).sum(dim=1)  # (B, N)
        scores = (dot_products > 0.9).sum(dim=1)  # (B,)

        return scores, hypothesis

    def ransac(self):

        all_hypo = []
        all_scores = []
        for _ in range(self.num_iter):
            hypo_points = self.batched_hypothesis()
            scores, hypo_points = self.scores(hypo_points)
            all_hypo.append(hypo_points)
            all_scores.append(scores)

        all_hypo = torch.stack(all_hypo, dim=0)  # (I, B, 2)
        all_scores = torch.stack(all_scores, dim=0)  # (I, B)

        # best_scores, best_idx = all_scores.max(dim=0) 
        # final_keypoints = all_hypo[best_idx, torch.arange(self.batch_size)]

        # Convert scores to weights and use them to compute 
        # a weighted average of the hypotheses to get the final keypoints
        weights = torch.softmax(all_scores.float(), dim=0) # (Iter, B)
        final_keypoints = (all_hypo * weights.unsqueeze(2)).sum(dim=0) # (B, 2)

        return final_keypoints

