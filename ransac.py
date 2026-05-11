import torch


class PVNetRansac:
    def __init__(self, mask: torch.Tensor, vfield: torch.Tensor, num_iter: int):

        self.num_keypoints = vfield.size(0) // 2
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
        self.vfield = torch.stack([x_components, y_components], dim=1)  # K, 2, H, W

        # mask is expected to be a binary foreground mask with shape (H, W).
        # valid index shape is (N, 2) where each row is (row, col).
        self.valid_index = mask.nonzero()  # N, 2  (row, col)

        # vfield shape is (K, 2, H, W)
        # Pre-compute used in self.scores
        self.mask_vecs = self.vfield[
            :, :, self.valid_index[:, 0], self.valid_index[:, 1]
        ]  # (K, 2, N)
        # flip valid index to x, y, transpose to 2, N
        self.mask_coords = self.valid_index.flip(1).float().T  # (2, N)

    def _sample_pairs(self, num_samples: int):
        num_valid = self.valid_index.size(0)

        first = torch.randint(
            num_valid,
            (num_samples,),
            device=self.valid_index.device,
        )
        second = torch.randint(
            num_valid - 1,
            (num_samples,),
            device=self.valid_index.device,
        )
        second = second + (second >= first).to(second.dtype)

        return self.valid_index[first], self.valid_index[second]

    def batched_hypothesis(self, num_samples: int = 1):
        """Batch hypothesis over K keypoints and optionally over multiple iterations."""

        assert self.vfield.dim() == 4 and self.vfield.size(1) == 2  # (K, 2, H, W)
        assert self.valid_index.dim() == 2 and self.valid_index.size(1) == 2  # (N, 2)

        # If the mask valid pixel is too small, return NaN hypotheses which will be ignored in scoring
        if self.valid_index.size(0) < 2:
            shape = (
                (self.num_keypoints, 2)
                if num_samples == 1
                else (num_samples, self.num_keypoints, 2)
            )
            return torch.full(shape, float("nan"), device=self.vfield.device)

        p1, p2 = self._sample_pairs(num_samples)

        v1 = self.vfield[:, :, p1[:, 0], p1[:, 1]].permute(2, 0, 1)  # (I, K, 2)
        v2 = self.vfield[:, :, p2[:, 0], p2[:, 1]].permute(2, 0, 1)  # (I, K, 2)

        assert v1.dim() == 3
        assert v1.size(0) == num_samples
        assert v1.size(1) == self.num_keypoints
        assert v1.size(2) == 2

        # Convert sampled pixel coords from tensor order (row, col) = (y, x) to geometric (x, y)
        p1 = p1.flip(1).float()  # (I, 2)
        p2 = p2.flip(1).float()  # (I, 2)

        assert p1.dim() == 2
        assert p1.size(0) == num_samples
        assert p1.size(1) == 2

        # Vectorized intersect line
        A = torch.stack([v1, -v2], dim=-1)  # (I, K, 2, 2)
        b = (p2 - p1).unsqueeze(1).expand(-1, self.num_keypoints, -1).float()  # from (I, 2) to (I, K, 2)

        assert A.dim() == 4 and A.size(2) == 2 and A.size(3) == 2
        assert b.dim() == 3 and b.size(2) == 2

        # Detect near-singular systems (parallel vectors)
        det = A[:, :, 0, 0] * A[:, :, 1, 1] - A[:, :, 0, 1] * A[:, :, 1, 0]  # (I, K)
        singular = det.abs() < 1e-6

        # Cramer's rule for 2x2 (avoids linalg.solve exceptions on singular)
        safe_det = det.clone()
        safe_det[singular] = 1.0  # avoid div-by-zero; result will be invalidated
        t0 = (
            A[:, :, 1, 1] * b[:, :, 0] - A[:, :, 0, 1] * b[:, :, 1]
        ) / safe_det  # (I, K)

        hypothesis = p1.unsqueeze(1) + t0.unsqueeze(2) * v1  # (I, K, 2)

        # Mark singular as NaN
        hypothesis[singular] = float("nan")

        if num_samples == 1:
            return hypothesis[0]

        return hypothesis

    def scores(self, hypothesis: torch.Tensor):

        if hypothesis.dim() == 2:
            assert hypothesis.size(0) == self.num_keypoints and hypothesis.size(1) == 2

            dir_to_hypo = (
                hypothesis[:, :, None] - self.mask_coords[None, :, :]
            )  # (K, 2, N)
            dir_to_hypo = torch.nn.functional.normalize(dir_to_hypo, dim=1)  # (K, 2, N)
            dot_products = (dir_to_hypo * self.mask_vecs).sum(dim=1)  # (K, N)
            scores = (dot_products > 0.99).sum(dim=1)  # (K,)
            scores[hypothesis.isnan().any(dim=1)] = 0
            return scores, hypothesis

        assert hypothesis.dim() == 3
        assert hypothesis.size(1) == self.num_keypoints and hypothesis.size(2) == 2

        dir_to_hypo = (
            hypothesis[:, :, :, None] - self.mask_coords[None, None, :, :]
        )  # (I, K, 2, N)
        dir_to_hypo = torch.nn.functional.normalize(dir_to_hypo, dim=2)  # (I, K, 2, N)

        dot_products = (dir_to_hypo * self.mask_vecs[None, :, :, :]).sum(
            dim=2
        )  # (I, K, N)

        # NaN hypotheses → 0 inliers
        scores = (dot_products > 0.99).sum(dim=2)  # (I, K)
        scores[hypothesis.isnan().any(dim=2)] = 0

        return scores, hypothesis

    def ransac(self):
        # Where I is the number of iterations and K is the number of keypoints.
        all_hypo = self.batched_hypothesis(num_samples=self.num_iter)  # (I, K, 2)
        all_scores, all_hypo = self.scores(all_hypo)  # (I, K), (I, K, 2)

        # # Pick the hypothesis with highest consensus per keypoint
        # best_idx = all_scores.argmax(dim=0)  # (K,)
        # final_keypoints = all_hypo[best_idx, torch.arange(self.num_keypoints)]  # (K, 2)

        # Filter NaN hypotheses before weighted average
        # By setting NaN hypo to (0,0) and score to 0
        valid_mask = torch.isfinite(all_hypo).all(dim=2)  # (I, K)
        safe_scores = all_scores * valid_mask.to(all_scores.dtype)
        safe_hypo = torch.where(
            valid_mask.unsqueeze(2), all_hypo, torch.zeros_like(all_hypo)
        )

        # Softmax weigthted average
        # Penalize low-score hypotheses by setting their weight close to zero
        weights = torch.softmax(safe_scores.float(), dim=0)  # (Iter, K)
        final_keypoints = (safe_hypo * weights.unsqueeze(2)).sum(dim=0)  # (K, 2)

        return final_keypoints
