import numpy as np
import cv2


def pad16(img):
    h, w = img.shape[:2]
    ph, pw = (-h) % 16, (-w) % 16
    return np.pad(img, ((0, ph), (0, pw)) + ((0, 0),) * (img.ndim - 2))

def visualiser(window, OMS, saliency_map, salmax_coords, panel_size=200):
    """
    Trimmed-down version of demo.py's visualiser(): drops the control-loop
    panel (cmd / pan-tilt target dot) since there is no controller here.
    Shows: raw events | saliency map | OMS map, with the saliency peak marked.

    Each input array (window, saliency_map, OMS) may come out at a *different*
    resolution depending on the source data (unlike demo.py's fixed 128x128
    sensor, here they depend on whatever resolution the .npy file has and how
    egomotion()/run_attentiondemo() reshape it). So every panel is explicitly
    resized to `panel_size` x `panel_size` before being combined -- this
    normalises the display and also fixes the salmax_coords marker position,
    which is rescaled into the same panel coordinates.
    """
    h = w = panel_size
    combined = np.full((h + 20, w * 3 + 4, 3), fill_value=255, dtype=np.uint8)

    def to_rgb(img, orig_shape=None):
        img = np.asarray(img, dtype=np.float32)

        img_min, img_max = img.min(), img.max()

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255
        else:
            img = np.zeros_like(img)

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img[:, :, None].repeat(3, axis=2)

    combined[10:10 + h, 1:1 + w, :] = to_rgb(window)
    combined[10:10 + h, w + 2:w + 2 + w, :] = to_rgb(OMS)
    combined[10:10 + h, 2 * w + 3:2 * w + 3 + w, :] = to_rgb(saliency_map)

    # salmax_coords are in saliency_map's original coordinate space --
    # rescale them into the resized panel before drawing
    sal_h, sal_w = np.asarray(saliency_map).shape
    scaled_y = int(salmax_coords[0] * h / sal_h)
    scaled_x = int(salmax_coords[1] * w / sal_w)
    cv2.circle(
        combined,
        (2 * w + scaled_x, 10 + scaled_y),
        4, (0, 255, 0), -1
    )

    cv2.putText(combined, 'Events', (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(combined, 'OMS', (w + 7, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(combined, 'Saliency', (2 * w + 8, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return pad16(combined)
