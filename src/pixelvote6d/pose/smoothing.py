import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PoseSmoother:
    def __init__(self, alpha_rvec: float = 0.5, alpha_tvec: float = 0.5):
        self.alpha_rvec = alpha_rvec
        self.alpha_tvec = alpha_tvec
        self.prev_rot = None
        self.prev_tvec = None

    def update(self, rvec: np.ndarray, tvec: np.ndarray):

        assert rvec.shape == (3, 1) and tvec.shape == (3, 1)

        curr_tvec = tvec.reshape(3)
        curr_rot = Rotation.from_rotvec(rvec.reshape(3))

        if self.prev_rot is None or self.prev_tvec is None:
            self.prev_rot = curr_rot
            self.prev_tvec = curr_tvec.copy()
            return rvec, tvec

        smooth_tvec = self._EMA_tvec(curr_tvec)

        key_times = [0.0, 1.0]
        key_rots = Rotation.concatenate([self.prev_rot, curr_rot])
        slerp = Slerp(key_times, key_rots)
        smooth_rot = slerp([self.alpha_rvec])[0]

        self.prev_tvec = smooth_tvec.copy()
        self.prev_rot = smooth_rot

        return smooth_rot.as_rotvec().reshape(3, 1), smooth_tvec.reshape(3, 1)

    def _EMA_tvec(self, current: np.ndarray):
        if self.prev_tvec is None:
            return current

        assert current.shape == self.prev_tvec.shape

        return self.alpha_tvec * current + (1 - self.alpha_tvec) * self.prev_tvec
