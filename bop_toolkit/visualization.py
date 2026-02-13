import cv2
import torch
from bop_toolkit.data_transfrom import PVNetTransform

def show_vfield(
    image: torch.Tensor,
    mask: torch.Tensor,
    vfield: torch.Tensor,
    keypoints_2d: torch.Tensor,
):

    mask = mask.detach().cpu().squeeze().numpy()  # (H, W)
    vfield = vfield.detach().cpu().squeeze().numpy()  # (16, H, W)
    kp2d = keypoints_2d.detach().cpu().squeeze().numpy()

    img = PVNetTransform.unnormalize_image(image.squeeze())
    img = img.permute(1, 2, 0).cpu().numpy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    num_keypoints = vfield.shape[0] // 2  # 8

    H, W = mask.shape

    print(
        f"num_keypoints: {num_keypoints}, image shape: {img.shape}, mask shape: {mask.shape}, vfield shape: {vfield.shape}"
    )
    for kp in kp2d:
        cv2.circle(img_bgr, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)

    for idx in range(num_keypoints):
        displayed_img = img_bgr.copy()

        for row in range(0, H, 20):
            for col in range(0, W, 20):
                if mask[row, col] > 0.5:
                    # Channel layout: [x0..x7, y0..y7]
                    vx = vfield[idx, row, col]  # x-component
                    vy = vfield[idx + num_keypoints, row, col]  # y-component

                    start_point = (col, row)  # (x, y) for OpenCV
                    end_point = (int(col + vx * 30), int(row + vy * 30))
                    cv2.arrowedLine(
                        displayed_img,
                        start_point,
                        end_point,
                        (0, 0, 255),
                        1,
                        tipLength=0.2,
                    )

        displayed_img = cv2.resize(
            displayed_img, (640, 480), interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow(f"Keypoint {idx}", displayed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()