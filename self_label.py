import torch
import cv2
import argparse
import numpy as np
import model
import ransac
from bop_toolkit.data_transfrom import PVNetTransformV2, PVNetTransform


parser = argparse.ArgumentParser(description="Visualize 3D pose")
parser.add_argument("--img", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output", type=str, required=True, default=None)
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()

# Setup model
device = "cuda" if torch.cuda.is_available() else "cpu"

pvnet = model.PVNet()
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
    mask = (
        (torch.sigmoid(pred_mask) > 0.5).cpu().numpy()
    )  # threshold on probability, not logit

    # mask_prob = torch.sigmoid(pred_mask).squeeze().cpu().numpy()

    if args.debug:
        # display = img.cpu().squeeze()
        # display = unnormalize(display)
        # cv2.imshow("Predicted Keypoints", display.permute(1, 2, 0).numpy())
        cv2.imshow("Predicted Keypoints", mask.astype(np.uint8) * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
