import torch
import pytest
from ransac import PVNetRansac
from utils.vector_field import generate_vector_field

SEED = 42


def generate_test_data(height=28, width=28):
    mask = torch.ones((height, width), dtype=torch.bool)
    keypoints = torch.tensor([[5, 5], [20, 5], [5, 20], [20, 20]], dtype=torch.float32)
    vector_field = generate_vector_field(height, width, mask.numpy(), keypoints.numpy())
    vfield = torch.from_numpy(vector_field).float()

    return mask, vfield


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_construct():
    NUM_KEYPOINTS = 8
    H, W = 28, 28
    dummy_mask = torch.ones((H, W), dtype=torch.bool)
    data = torch.full((NUM_KEYPOINTS * 2, H, W), 1.0)
    pvnet_ransac = PVNetRansac(mask=dummy_mask, vfield=data, num_iter=10)

    assert pvnet_ransac.vfield.shape == (NUM_KEYPOINTS, 2, H, W)


def test_batched_hypothesis():
    NUM_KEYPOINTS = 4
    mask, vfield = generate_test_data()

    pvnet_ransac = PVNetRansac(mask=mask, vfield=vfield, num_iter=10)

    hypothesis = pvnet_ransac.batched_hypothesis()

    assert hypothesis.shape == (NUM_KEYPOINTS, 2)

def test_ransac():
    NUM_KEYPOINTS = 4
    mask, vfield = generate_test_data()

    pvnet_ransac = PVNetRansac(mask=mask, vfield=vfield, num_iter=10)

    keypoints = pvnet_ransac.ransac()

    assert keypoints.shape == (NUM_KEYPOINTS, 2)

