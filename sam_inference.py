import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

def generate_sam_masks(
    image_path,
    sam_processor,
    sam_model,
    grid_spacing=50
):
    # Load image for SAM
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [1024, 1024])
    img = tf.cast(img, tf.float32) / 255.0
    img_batch = tf.expand_dims(img, 0)

    # Original image
    orig_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_image.size

    # Grid points
    xs = np.arange(grid_spacing // 2, 1024, grid_spacing)
    ys = np.arange(grid_spacing // 2, 1024, grid_spacing)
    grid_points = [(x, y) for y in ys for x in xs]

    all_masks = []

    for x, y in grid_points:
        point = tf.constant([[[x, y]]], dtype=tf.float32)
        inputs = sam_processor(
            img_batch,
            input_points=point,
            return_tensors="tf",
            do_rescale=False
        )

        outputs = sam_model(**inputs)

        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
            return_tensors="tf"
        )[0]  # (1, 3, H, W)

        for i in range(masks.shape[1]):
            mask = (masks[0, i].numpy() > 0.5).astype(np.uint8)
            resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            all_masks.append(resized)

    return orig_image, all_masks
