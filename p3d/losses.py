import torch

import lpips
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel


loss_fn_alex = lpips.LPIPS(net='alex')

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dino = AutoModel.from_pretrained('facebook/dinov2-base')

CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def calc_l2_losses(ref, tests):
    """get l2 loss of tests compared with ref

    Args:
        ref (torch.tensor): reference image (h, w, 3)
        tests (torch.tensor): comparison images (batch, h, w, 3)
    """
    l2_losses = ((tests - ref.unsqueeze(0)) ** 2).mean(dim=[1, 2, 3])
    return l2_losses

def calc_lpips_losses(ref, tests):
    """get lpips loss of tests compared with ref

    Args:
        ref (torch.tensor): reference image (h, w, 3)
        tests (torch.tensor): comparison images (batch, h, w, 3)
    """

    lpips_losses = loss_fn_alex(tests.permute(0, 3, 1, 2)* 2 - 1, ref.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1)
    return lpips_losses

def calc_clip_losses(ref, tests):
    """get clip loss of tests compared with ref

    Args:
        ref (torch.tensor): reference image (h, w, 3)
        tests (torch.tensor): comparison images (batch, h, w, 3)
    """
    ref_features = CLIP.get_image_features(**clip_processor(images=ref, return_tensors="pt", do_rescale=False)).detach()
    test_features = CLIP.get_image_features(**clip_processor(images=tests, return_tensors="pt", do_rescale=False)).detach()
    
    clip_losses = 1 - torch.nn.functional.cosine_similarity(test_features, ref_features, dim=1).clamp(0, 1)
    return clip_losses

def calc_dino_losses(ref, tests):
    """get dino loss of tests compared with ref

    Args:
        ref (torch.tensor): reference image (h, w, 3)
        tests (torch.tensor): comparison images (batch, h, w, 3)
    """

    ref_features = dino(**processor(images=ref, return_tensors="pt", do_rescale=False)).pooler_output.detach()
    test_features = dino(**processor(images=tests, return_tensors="pt", do_rescale=False)).pooler_output.detach()

    dino_losses = 1 - torch.nn.functional.cosine_similarity(test_features, ref_features, dim=1).clamp(0, 1)
    return dino_losses