from contextlib import contextmanager
from typing import Callable
import torch
from ...modules.sparse.attention import SparseMultiHeadAttentionWithAttentionMap


class LocalAttentionState:
    def __init__(self, interval: int = 24):
        self.maps = []
        self.voxel_maps = []
        self.current_mean = None
        self.current_voxel_mean = None
        self.count = 0
        self.interval = interval

    def update(self, attn_scores: torch.Tensor):
        attn = torch.softmax(attn_scores, dim=-1)
        voxel_attn = torch.softmax(attn_scores, dim=-2)

        if self.current_mean is None:
            self.current_mean = attn
            self.current_voxel_mean = voxel_attn
            self.count = 1
        elif self.count % self.interval == 0:
            self.maps.append(self.current_mean)
            self.voxel_maps.append(self.current_voxel_mean)
            self.current_mean = attn
            self.current_voxel_mean = voxel_attn
            self.count = 1
        else:
            self.current_mean = (self.current_mean * self.count + attn) / (
                self.count + 1
            )
            self.current_voxel_mean = (
                self.current_voxel_mean * self.count + voxel_attn
            ) / (self.count + 1)
            self.count += 1

    def finalize(self):
        if self.current_mean is not None:
            self.maps.append(self.current_mean)
            self.voxel_maps.append(self.current_voxel_mean)


def build_local_attention_hook(attn_state: LocalAttentionState) -> Callable:
    def hook(module, inputs, outputs):
        output, attn_scores = outputs
        if attn_scores is None:
            return output
        attn_state.update(attn_scores)
        return output

    return hook


def drop_attention_hook(module, inputs, outputs):
    return outputs[0]


@contextmanager
def cross_attention_hook_context(
    flow_model,
    attention_hook_fn,
    drop_hook_fn,
    get_attention: bool = True,
):
    hook_handle_list = []
    original_cross_attns = []

    try:
        for module in flow_model.blocks:
            original_cross_attns.append(module.cross_attn)
            module.cross_attn = SparseMultiHeadAttentionWithAttentionMap(
                module.cross_attn
            )

            handle = module.cross_attn.register_forward_hook(
                attention_hook_fn if get_attention else drop_hook_fn
            )
            hook_handle_list.append(handle)

        yield

    finally:
        for handle in hook_handle_list:
            handle.remove()

        for module, original_attn in zip(flow_model.blocks, original_cross_attns):
            module.cross_attn = original_attn