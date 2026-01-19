import torch
from transformers import SamProcessor, TFSamModel, CLIPProcessor, CLIPModel

def load_sam():
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
    sam_model = TFSamModel.from_pretrained("facebook/sam-vit-large")
    return sam_processor, sam_model


def load_clip(device):
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    return clip_processor, clip_model

