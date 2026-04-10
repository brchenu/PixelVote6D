import torch


class PVNetRansac:
    def __init__(self, mask: torch.Tensor, vfield: torch.Tensor, num_iter: int):

        self.batch_size = self.num_keypoints
        self.num_iter = num_iter
        self.height = vfield.size(1)
        self.width = vfield.size(2)

        # vfield shape is (K*2, H, W)
        self.num_keypoints = vfield.size(0) // 2

        # In vfield x components are in the first half channels, y components in the second half
        x_components = vfield[: self.num_keypoints]
        y_components = vfield[self.num_keypoints :]

        # Convert to (K, 2, H, W) for easier indexing
        # vfield[k, 0] is x component, vfield[k, 1] is y component
        self.vfield = torch.stack([x_components, y_components], dim=1)  # B, 2, H, W

        # mask shape is (H, W)
        self.valid_index = (mask > 0.5).nonzero()  # N, 2  (row, col)

        # vfield shape is (B, 2, H, W)
        # Pre-compute used in self.scores
        self.mask_vecs = self.vfield[
            :, :, self.valid_index[:, 0], self.valid_index[:, 1]
        ]  # (B, 2, N)
        # flip valid index to x, y, transpose to 2, N
        self.mask_coords = self.valid_index.flip(1).float().T  # (2, N)

    def batched_hypothesis(self):
        """Batch hypothesis over B keypoints"""

        assert self.vfield.dim() == 4 and self.vfield.size(1) == 2  # B, 2, H, W
        assert self.valid_index.dim() == 2
        assert self.valid_index.size(1) == 2  # N, 2

        # If the mask valid pixel is too small, return NaN hypotheses which will be ignored in scoring
        if self.valid_index.size(0) < 2:
            return torch.full(
                (self.batch_size, 2), float("nan"), device=self.vfield.device
            )

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

        # Detect near-singular systems (parallel vectors)
        det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]  # (B,)
        singular = det.abs() < 1e-6

        # Cramer's rule for 2x2 (avoids linalg.solve exceptions on singular)
        safe_det = det.clone()
        safe_det[singular] = 1.0  # avoid div-by-zero; result will be invalidated
        t0 = (A[:, 1, 1] * b[:, 0] - A[:, 0, 1] * b[:, 1]) / safe_det  # (B,)

        hypothesis = p1 + t0.unsqueeze(1) * v1  # (B, 2)

        # Mark singular or out-of-bounds hypotheses as NaN
        oob = (
            (hypothesis[:, 0] < 0)
            | (hypothesis[:, 0] >= self.width)
            | (hypothesis[:, 1] < 0)
            | (hypothesis[:, 1] >= self.height)
        )
        hypothesis[singular | oob] = float("nan")

        return hypothesis

    def scores(self, hypothesis: torch.Tensor):

        assert hypothesis.size(0) == self.batch_size and hypothesis.size(1) == 2

        dir_to_hipo = hypothesis[:, :, None] - self.mask_coords[None, :, :]  # (B, 2, N)
        dir_to_hipo = torch.nn.functional.normalize(dir_to_hipo, dim=1)  # (B, 2, N)

        dot_products = (dir_to_hipo * self.mask_vecs).sum(dim=1)  # (B, N)

        # NaN hypotheses → 0 inliers
        scores = (dot_products > 0.99).sum(dim=1)  # (B,)
        scores[hypothesis.isnan().any(dim=1)] = 0

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

        # Pick the hypothesis with highest consensus per keypoint
        best_idx = all_scores.argmax(dim=0)  # (B,)
        final_keypoints = all_hypo[best_idx, torch.arange(self.batch_size)]  # (B, 2)

        # weights = torch.softmax(all_scores.float(), dim=0) # (Iter, B)
        # final_keypoints = (all_hypo * weights.unsqueeze(2)).sum(dim=0) # (B, 2)

        return final_keypoints
