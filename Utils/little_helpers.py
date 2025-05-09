from contextlib import contextmanager
import time
from typing import Optional, List
import os
import numpy as np
import torch
import torch.nn as nn


@contextmanager
def timer(name: str) -> None:
    """
    Context manager to measure execution time of code blocks.
    """
    start_time = time.time()
    yield
    elapsed_seconds = time.time() - start_time

    # Format time as hh:mm:ss
    hours, remainder = divmod(int(elapsed_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((elapsed_seconds - int(elapsed_seconds)) * 1000)

    if hours > 0:
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    print(f"{name} completed in {time_str}")


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    else:
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_metrics(train_losses: List[float], train_accs: List[float],
                 test_losses: List[float], test_accs: List[float],
                 output_dir: str) -> None:

    metrics = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'test_loss': test_losses,
        'test_acc': test_accs
    }

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'metrics.npy'), metrics)


def load_pretrained_vit(model: nn.Module,
                        pretrained_model_name: str = "vit_base_patch32_224"):
    """
    Load pretrained weights from timm into your custom ViT model.
    Only works for ViTs with the same patch size and number of patches
    """
    import timm

    # Load pretrained model
    pretrained_model = timm.create_model(pretrained_model_name, pretrained=True)
    pretrained_dict = pretrained_model.state_dict()

    # Filter out unnecessary keys
    model_dict = model.state_dict()

    # 1. Filter out keys that don't match in shape
    filtered_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    # 2. Handle patch embedding specifically
    if 'patch_embed.proj.weight' in model_dict and 'patch_embed.proj.weight' in pretrained_dict:
        if model_dict['patch_embed.proj.weight'].shape != pretrained_dict['patch_embed.proj.weight'].shape:
            print(f"Skipping patch_embed.proj.weight due to shape mismatch: "
                  f"{model_dict['patch_embed.proj.weight'].shape} vs {pretrained_dict['patch_embed.proj.weight'].shape}")

    # 3. Update model state dict
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)

    print(f"Loaded {len(filtered_dict)} / {len(pretrained_dict)} layers from pretrained model")

    return model

def get_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    return parameters
    


def get_overlap(a, b):
    """Returns proportion overlap between two boxes in the form (tl, width, height, confidence, class)."""

    a_tl, a_width, a_height, _, _ = a
    b_tl, b_width, b_height, _, _ = b
    i_tl = (
        max(a_tl[0], b_tl[0]),
        max(a_tl[1], b_tl[1])
    )
    i_br = (
        min(a_tl[0] + a_width, b_tl[0] + b_width),
        min(a_tl[1] + a_height, b_tl[1] + b_height),
    )

    intersection = max(0, i_br[0] - i_tl[0]) * max(0, i_br[1] - i_tl[1])
    
    a_area = a_width * a_height
    b_area = b_width * b_height

    
    a_intersection = b_intersection = intersection

    union = (a_area + b_area - intersection)
    
    ## incase there is no bbbx
    if a_area == 0:
        a_intersection = torch.tensor(0)
        a_area = 0.2
    if b_area == 0:
        b_intersection = torch.tensor(0)
        b_area = 0.2

    intersection = torch.tensor(intersection)
    
    return (intersection/float(union)).item()

def get_bboxes(m,n,labels, grid_size_x, grid_size_y,num_classes, min_confidence, image_size):

    bboxes = []
    for i in range(m):
        for j in range(n):
            for k in range((labels.size(dim=2) - num_classes) // 5):
                bbox_start = 5 * k + num_classes
                bbox_end = 5 * (k + 1) + num_classes
                bbox = labels[i, j, bbox_start:bbox_end]
                class_index = torch.argmax(labels[i, j, :num_classes]).item()
                confidence = labels[i, j, class_index].item() * bbox[4].item()          # pr(c) * IOU
                #print(confidence)
                if confidence > min_confidence:
                    width = bbox[2] * image_size[0]
                    height = bbox[3] * image_size[1]
                    tl = (
                        bbox[0] * image_size[0] + j * grid_size_x - width / 2,
                        bbox[1] * image_size[1] + i * grid_size_y - height / 2
                    )
                    bboxes.append([tl, width, height, confidence, class_index])

    return(bboxes)

