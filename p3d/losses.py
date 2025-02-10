import torch

from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
import lpips


loss_fn_alex = lpips.LPIPS(net='alex')

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