import os
import torch
import torch.nn as nn
import huggingface_hub
from xtuner.registry import BUILDER

try:
    from safetensors.torch import load_file as st_load_file, save_file as st_save_file
    _HAS_ST = True
except Exception:
    _HAS_ST = False

def is_dist():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def is_rank0():
    return (not is_dist()) or torch.distributed.get_rank() == 0


@BUILDER.register_module()
def create_longvit_model_fast(
    model_arch: str,
    in_chans: int,
    *,
    hub_id: str | None = None,          # e.g. "prov-gigapath/prov-gigapath"
    local_ckpt: str | None = None,      # e.g. "/path/slide_encoder.(safetensors|pth)"
    filename: str = "slide_encoder.pth", # HF filename
    local_dir: str | None = None,
    global_pool=False,
    tile_size=224,
    max_wsi_size=420096,
    token_norm = True,
    dtype=None,
):
    assert (hub_id is None) ^ (local_ckpt is None), \
        "Provide exactly one of {hub_id, local_ckpt}"

    from xtuner.model.utils import LoadWoInit
    from xtuner.model.torchscale.model.LongNetVit import create_model_original as _create
    with LoadWoInit():
        model = _create(
            pretrained="", model_arch=model_arch, 
            in_chans=in_chans,
            global_pool=global_pool,
            tile_size=tile_size,
            max_wsi_size=max_wsi_size,
            token_norm = token_norm
        )

    # --- resolve checkpoint path ---
    if local_ckpt is not None:
        ckpt_path = local_ckpt
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[LongNet] local_ckpt not found: {ckpt_path}")
    else:
        # hub path
        if is_rank0():
            huggingface_hub.hf_hub_download(
                hub_id, filename=filename,
                local_dir=local_dir or os.path.join(os.path.expanduser("~"), ".cache"),
                force_download=False, local_files_only=False
            )
        if is_dist():
            torch.distributed.barrier()
        ckpt_path = os.path.join(local_dir or os.path.join(os.path.expanduser("~"), ".cache"), filename)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[LongNet] HF file not present after download: {ckpt_path}")

    # --- prefer safetensors if provided ---
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file as st_load
        sd = st_load(ckpt_path, device="cpu")
    else:
        try:
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(ckpt_path, map_location="cpu")
        sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    # --- void-copy first layer; filter mismatches ---
    sd.pop("patch_embed.proj.weight", None)
    sd.pop("patch_embed.proj.bias", None)
    msd = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(sd, strict=False)

    # --- reinit first layer + sync dtype/device ---
    m = model.patch_embed.proj
    torch.nn.init.xavier_uniform_(m.weight); 
    if m.bias is not None: torch.nn.init.zeros_(m.bias)
    # ref = next(p for p in model.parameters() if p.is_floating_point())
    # m.to(device=ref.device, dtype=ref.dtype)

    if dtype is not None:
        model = model.to(dtype=dtype)

    # if is_rank0():
    print(f"[LongNet] loaded from {'hub:'+hub_id if hub_id else ckpt_path}")
    return model