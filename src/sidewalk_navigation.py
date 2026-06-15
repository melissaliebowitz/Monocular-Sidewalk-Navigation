import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from depth_anything_v2.dpt import DepthAnythingV2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}

CHECKPOINT_PATH = "checkpoints/depth_anything_v2_vits.pth"
SEG_MODEL_PATH = "segformer_sidewalk_road_model"

SIDEWALK_CLASS_ID = 1
ROAD_CLASS_ID = 2


def load_depth_model():
    model = DepthAnythingV2(**MODEL_CONFIG)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model


def load_segmentation_model():
    processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_PATH)
    model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_PATH)
    model = model.to(DEVICE).eval()
    return processor, model


def normalize_depth(depth):
    depth = depth.astype(np.float32)
    d_min, d_max = depth.min(), depth.max()

    if d_max - d_min < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)

    depth_norm = (depth - d_min) / (d_max - d_min)
    return (depth_norm * 255).astype(np.uint8)


def get_segmentation_mask(frame, processor, seg_model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = seg_model(**inputs)

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=frame.shape[:2],
        mode="bilinear",
        align_corners=False,
    )

    predicted = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    sidewalk_mask = predicted == SIDEWALK_CLASS_ID
    road_mask = predicted == ROAD_CLASS_ID

    sidewalk_mask = sidewalk_mask.astype(np.uint8) * 255

    kernel = np.ones((7, 7), np.uint8)
    sidewalk_mask = cv2.morphologyEx(sidewalk_mask, cv2.MORPH_CLOSE, kernel)
    sidewalk_mask = cv2.morphologyEx(sidewalk_mask, cv2.MORPH_OPEN, kernel)

    return sidewalk_mask, predicted


def analyze_navigation(frame, depth, sidewalk_mask):
    h, w = depth.shape

    y1 = int(h * 0.45)
    y2 = int(h * 0.95)

    # Keep only lower-half sidewalk area
    tracked_mask = np.zeros_like(sidewalk_mask)
    tracked_mask[y1:y2, :] = sidewalk_mask[y1:y2, :]

    # Find connected sidewalk region
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tracked_mask)

    target_label = 0
    largest_area = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            target_label = label

    if target_label > 0:
        tracked_mask = (labels == target_label).astype(np.uint8) * 255
    else:
        tracked_mask = np.zeros_like(sidewalk_mask)

    left_boundary = np.zeros(h, dtype=np.int32)
    right_boundary = np.ones(h, dtype=np.int32) * w

    for y in range(y1, y2):
        row_pixels = np.where(tracked_mask[y, :] > 0)[0]

        if len(row_pixels) > 0:
            left_boundary[y] = row_pixels[0]
            right_boundary[y] = row_pixels[-1]
        elif y > y1:
            left_boundary[y] = left_boundary[y - 1]
            right_boundary[y] = right_boundary[y - 1]

    masked_depth = depth.copy()

    for y in range(y1, y2):
        l_x = left_boundary[y]
        r_x = right_boundary[y]

        masked_depth[y, :l_x] = 255
        masked_depth[y, r_x:] = 255

    masked_depth[y2:, :] = 255

    obstacle_y_end = int(h * 0.85)

    left_scores = []
    center_scores = []
    right_scores = []

    for y in range(y1, obstacle_y_end):
        l_x = left_boundary[y]
        r_x = right_boundary[y]
        sidewalk_width = r_x - l_x

        if sidewalk_width > 30:
            seg_w = sidewalk_width / 3.0

            lx1 = int(l_x)
            lx2 = int(l_x + seg_w)
            cx1 = lx2
            cx2 = int(l_x + 2 * seg_w)
            rx1 = cx2
            rx2 = int(r_x)

            left_scores.append(np.mean(masked_depth[y, lx1:lx2]))
            center_scores.append(np.mean(masked_depth[y, cx1:cx2]))
            right_scores.append(np.mean(masked_depth[y, rx1:rx2]))

    if len(left_scores) > 0:
        l_score = np.mean(left_scores)
        c_score = np.mean(center_scores)
        r_score = np.mean(right_scores)
    else:
        l_score, c_score, r_score = 255, 255, 255

    all_scores = np.array([l_score, c_score, r_score])
    threshold = max(100.0, np.mean(all_scores) + 0.25 * np.std(all_scores))

    obstacle_center = c_score > threshold

    if target_label == 0:
        instruction = "No sidewalk detected - STOP"
    elif obstacle_center:
        if l_score < r_score:
            instruction = "Obstacle ahead - move left"
        else:
            instruction = "Obstacle ahead - move right"
    else:
        instruction = "Path clear - continue forward"

    scores = (l_score, c_score, r_score, threshold)

    return instruction, scores, (y1, y2), tracked_mask, left_boundary, right_boundary, masked_depth


def draw_overlay(frame, instruction, scores, walk_area, tracked_mask, left_boundary, right_boundary):
    h, w, _ = frame.shape
    y1, y2 = walk_area
    l_score, c_score, r_score, threshold = scores

    overlay = frame.copy()

    # Green = detected sidewalk
    overlay[tracked_mask > 0] = overlay[tracked_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5

    for y in range(y1, y2, 2):
        cv2.circle(overlay, (left_boundary[y], y), 2, (0, 255, 255), -1)
        cv2.circle(overlay, (right_boundary[y], y), 2, (0, 255, 255), -1)

    cv2.putText(
        overlay,
        instruction,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
    )

    cv2.putText(
        overlay,
        f"Scores L/C/R: {l_score:.1f}, {c_score:.1f}, {r_score:.1f} Th:{threshold:.1f}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return overlay


def draw_segmentation_debug(frame, predicted):
    debug = frame.copy()

    sidewalk_mask = predicted == SIDEWALK_CLASS_ID
    road_mask = predicted == ROAD_CLASS_ID

    debug[sidewalk_mask] = (0, 255, 0)
    debug[road_mask] = (255, 0, 0)

    return cv2.addWeighted(frame, 0.6, debug, 0.4, 0)


def main():
    depth_model = load_depth_model()
    seg_processor, seg_model = load_segmentation_model()

    video = "data/sidewalk_video4.mp4"
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Could not open video.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))

        with torch.no_grad():
            depth = depth_model.infer_image(frame)

        depth_norm = normalize_depth(depth)

        sidewalk_mask, predicted = get_segmentation_mask(
            frame,
            seg_processor,
            seg_model
        )

        instruction, scores, walk_area, tracked_mask, left_boundary, right_boundary, masked_depth = analyze_navigation(
            frame,
            depth_norm,
            sidewalk_mask
        )

        output_frame = draw_overlay(
            frame.copy(),
            instruction,
            scores,
            walk_area,
            tracked_mask,
            left_boundary,
            right_boundary
        )

        depth_colormap = cv2.applyColorMap(masked_depth, cv2.COLORMAP_INFERNO)
        seg_debug = draw_segmentation_debug(frame.copy(), predicted)

        cv2.imshow("Navigation View", output_frame)
        cv2.imshow("Depth Map", depth_colormap)
        cv2.imshow("Segmentation Debug", seg_debug)

        print(instruction)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()