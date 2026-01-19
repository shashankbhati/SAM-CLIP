import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from .utils import crop_to_mask, categorize_masks_by_area


def filter_masks_with_clip(
    orig_image,
    masks,
    text_prompt,
    clip_processor,
    clip_model,
    device,
    clip_threshold=35,
    size_mode="mixed",
    preview=False
):
    selected_ids = categorize_masks_by_area(masks, size_mode)

    final_masks = []
    scores = []

    for idx in selected_ids:
        mask = masks[idx]

        img_np = np.array(orig_image)
        masked = np.zeros_like(img_np)
        masked[mask == 1] = img_np[mask == 1]
        masked_img = Image.fromarray(masked)

        cropped = crop_to_mask(masked_img, mask)
        if cropped is None or cropped.size[0] < 20 or cropped.size[1] < 20:
            continue

        inputs = clip_processor(
            text=[text_prompt],
            images=cropped,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)

        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        score = (img_emb @ txt_emb.T)[0][0].item() * 100

        if score > clip_threshold:
            final_masks.append(mask)
            scores.append(score)

            if preview:
                plt.imshow(cropped)
                plt.title(f"CLIP score: {score:.1f}")
                plt.axis("off")
                plt.show()

    # Overlay result
    overlay = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for mask in final_masks:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        draw.bitmap((0, 0), mask_img, fill=(255, 0, 0, 128))

    result = Image.alpha_composite(orig_image.convert("RGBA"), overlay)
    return result, final_masks, scores
