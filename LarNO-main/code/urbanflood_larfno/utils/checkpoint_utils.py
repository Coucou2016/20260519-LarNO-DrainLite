from __future__ import annotations

from collections import OrderedDict

import torch


def load_state_dict_with_optional_lifting_expand(
    model,
    checkpoint_path,
    map_location="cpu",
    expand_lifting=False,
    old_data_channels=13,
):
    """Load a checkpoint, optionally expanding the first lifting layer.

    LarNO-D adds static drainage channels to the original 13-channel input.
    All operator layers can still reuse the pretrained weights, but the first
    lifting Conv1d has a wider input dimension. For that one tensor, copy the
    original data channels and positional-grid channels into the new tensor and
    initialise the inserted drainage-channel weights to zero.
    """
    raw_state = torch.load(checkpoint_path, map_location=map_location)
    source_state = OrderedDict(
        (k[7:] if k.startswith("module.") else k, v)
        for k, v in raw_state.items()
    )
    target_state = model.state_dict()

    loaded = []
    expanded = []
    skipped = []

    for key, value in source_state.items():
        if key not in target_state:
            skipped.append((key, "missing-in-model"))
            continue

        target_value = target_state[key]
        if tuple(value.shape) == tuple(target_value.shape):
            target_state[key] = value
            loaded.append(key)
            continue

        if expand_lifting and key == "lifting.fcs.0.weight" and value.ndim == 3:
            try:
                target_state[key] = _expand_lifting_weight(
                    value, target_value, old_data_channels=old_data_channels
                )
                expanded.append(key)
                continue
            except ValueError as exc:
                skipped.append((key, str(exc)))
                continue

        skipped.append((key, f"shape {tuple(value.shape)} -> {tuple(target_value.shape)}"))

    model.load_state_dict(target_state)
    return {"loaded": loaded, "expanded": expanded, "skipped": skipped}


def _expand_lifting_weight(source, target, old_data_channels):
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("lifting weight must be Conv1d-shaped")
    if source.shape[0] != target.shape[0] or source.shape[2] != target.shape[2]:
        raise ValueError(
            f"incompatible lifting output/kernel shape {tuple(source.shape)} -> {tuple(target.shape)}"
        )
    old_in = source.shape[1]
    new_in = target.shape[1]
    if new_in <= old_in:
        raise ValueError("target lifting layer is not wider than source")
    old_grid_channels = old_in - int(old_data_channels)
    if old_grid_channels < 0:
        raise ValueError(
            f"old_data_channels={old_data_channels} exceeds source input channels={old_in}"
        )
    new_data_channels = new_in - old_grid_channels
    if new_data_channels < old_data_channels:
        raise ValueError("target has insufficient data channels")

    expanded = target.clone()
    expanded.zero_()
    expanded[:, :old_data_channels, :] = source[:, :old_data_channels, :]
    if old_grid_channels:
        expanded[:, new_data_channels:new_data_channels + old_grid_channels, :] = (
            source[:, old_data_channels:old_in, :]
        )
    return expanded
