from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

def get_device() -> "torch.device":
    """
    Get the device to use for torch operations.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_and_dtype_from_module(module: "torch.nn.Module") -> tuple["torch.device", "torch.dtype"]:
    """
    Get the device and dtype to use for a module.
    """
    if not hasattr(module, "device") or not hasattr(module, "dtype"):
        next_param = next(module.parameters())
        device = getattr(module, "device", next_param.device)
        dtype = getattr(module, "dtype", next_param.dtype)
    else:
        device = module.device
        dtype = module.dtype

    return device, dtype