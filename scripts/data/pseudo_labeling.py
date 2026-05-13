import cv2
import torch
import argparse

from pixelvote6d.models import PVNet
from pixelvote6d.pose import PVNetRansac
from pixelvote6d.dataset import PVNetTransformV2, PVNetTransform

parser = argparse.ArgumentParser(description="Visualize 3D pose")
parser.add_argument("--img", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)

args = parser.parse_args()

# Setup model
device = "cuda" if torch.cuda.is_available() else "cpu"

pvnet = PVNet()
pvnet.load_state_dict(
    torch.load(args.checkpoint, weights_only=True)["model_state_dict"]
)
pvnet.to(device)
pvnet.eval()

# Load image
transform = PVNetTransformV2().transform
unnormalize = PVNetTransform().unnormalize_image

img = cv2.imread(args.img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # model expects RGB
img = transform(img)

# Run inference
with torch.no_grad():
    img = img.unsqueeze(0).to(device)  # Add batch dimension
    pred_mask, pred_kp = pvnet(img)

    pred_mask = pred_mask.squeeze()
    prob_mask = (torch.sigmoid(pred_mask) > 0.5).cpu().numpy()
    binary_mask = torch.from_numpy(prob_mask).to(
        device=pred_mask.device, dtype=torch.float32
    )

    ransac_solver = PVNetRansac(
        mask=binary_mask, vfield=pred_kp.squeeze(), num_iter=1000
    )
    keypoints = ransac_solver.ransac()

    display = img.cpu().squeeze()
    display = unnormalize(display)  # (3, H, W) uint8 tensor

    # (H, W, 3) uint8 numpy now opencv compatible
    display = display.permute(1, 2, 0).numpy()
    display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)  # fix colors for imshow

    for x, y in keypoints.cpu().numpy():
        cv2.circle(display, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)

    display = cv2.resize(display, (640, 640), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Predicted Keypoints", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
