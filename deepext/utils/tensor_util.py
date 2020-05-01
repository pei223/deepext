import torch


def try_cuda(e):
    if torch.cuda.is_available() and hasattr(e, "cuda"):
        return e.cuda()
    return e


def parameter_count(model: torch.nn.Module):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params
