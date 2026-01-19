import torch
from sam_clip.models import load_sam, load_clip
from sam_clip.sam_inference import generate_sam_masks
from sam_clip.clip_filtering import filter_masks_with_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam_processor, sam_model = load_sam()
clip_processor, clip_model = load_clip(device)

image_path = "cat.jpeg"
prompt = "dog"

orig_img, masks = generate_sam_masks(
    image_path,
    sam_processor,
    sam_model,
    grid_spacing=50
)

result, final_masks, scores = filter_masks_with_clip(
    orig_img,
    masks,
    text_prompt=prompt,
    clip_processor=clip_processor,
    clip_model=clip_model,
    device=device,
    clip_threshold=35,
    size_mode="s",
    preview=True
)

result.show()
