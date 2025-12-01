from typing import List, Tuple
import torch
import numpy as np
from PIL import Image


def fix_mask_by_majority_tensor(
    points: torch.Tensor,
    selected_indices: torch.Tensor,
    k: int = 16,
    vote_ratio: float = 0.6,
) -> torch.Tensor:
    points_f = points.float()
    N = points.size(0)

    mask = torch.zeros(N, dtype=torch.bool, device=points.device)
    mask[selected_indices] = True

    dist = torch.cdist(points_f, points_f)
    knn_indices = dist.topk(k=k, largest=False)[1]

    neigh_mask = mask[knn_indices]
    neigh_mask_sum = neigh_mask.sum(dim=1)

    new_mask = mask.clone()

    fill_cond = (~mask) & (neigh_mask_sum >= k * vote_ratio)
    new_mask[fill_cond] = True

    clear_cond = mask & (neigh_mask_sum < k * (1 - vote_ratio))
    new_mask[clear_cond] = False

    return torch.nonzero(new_mask, as_tuple=False).view(-1)


def downsample_coords(
    coords: torch.Tensor,
    factor: Tuple[int, int, int],
    require_upsample: bool = False,
):
    DIM = coords.shape[-1] - 1
    assert DIM == len(
        factor
    ), "Input coordinates must have the same dimension as the downsample factor."

    coord = list(coords.unbind(dim=-1))
    for i, f in enumerate(factor):
        coord[i + 1] = coord[i + 1] // f

    MAX = [coord[i + 1].max().item() + 1 for i in range(DIM)]
    OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
    code = sum([c * o for c, o in zip(coord, OFFSET)])
    code, idx = code.unique(return_inverse=True)
    new_coords = torch.stack(
        [code // OFFSET[0]] + [(code // OFFSET[i + 1]) % MAX[i] for i in range(DIM)],
        dim=-1,
    )
    if not require_upsample:
        return new_coords
    else:
        return new_coords, idx


def get_valid_patches_from_images(images: List[Image.Image]) -> List[List[int]]:
    patch_size = 14
    valid_tokens_list: List[List[int]] = []
    for image in images:
        image_np = np.array(image)
        mask = image_np.sum(-1) != 0
        H, W = mask.shape
        h_blocks = H // patch_size
        w_blocks = W // patch_size

        mask_cropped = mask[: h_blocks * patch_size, : w_blocks * patch_size]
        mask_patches = mask_cropped.reshape(
            h_blocks, patch_size, w_blocks, patch_size
        )
        mask_patches = np.any(mask_patches, axis=(1, 3))
        selected_patches = mask_patches.flatten()
        selected_index = np.nonzero(selected_patches)[0].tolist()
        valid_tokens_list.append(selected_index)
    return valid_tokens_list


def exclude_patches_by_voxel_mask_attn(
    reference_image: Image.Image,
    attn_map: torch.Tensor,
    keep_voxel_index: torch.Tensor,
    ori_cond: dict,
    additional_tokens_num: int = 5,
    heads=None,
):
    if heads is None:
        heads = [0, 4, 12]

    attn_map = attn_map[heads].mean(0)
    valid_tokens = get_valid_patches_from_images([reference_image])[0]
    addition_valid_tokens = [x + additional_tokens_num for x in valid_tokens]
    attn_map = attn_map[keep_voxel_index][:, addition_valid_tokens].sum(0)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    attn_valid_image_indices = torch.nonzero(attn_map > 0.55).squeeze(1)
    attn_image_indices = [valid_tokens[i] for i in attn_valid_image_indices]
    attn_tokens_indices = list(range(additional_tokens_num)) + [
        additional_tokens_num + x for x in attn_image_indices
    ]
    ori_cond["cond"] = ori_cond["cond"][:, attn_tokens_indices]
    ori_cond["neg_cond"] = ori_cond["neg_cond"][:, attn_tokens_indices]
    return ori_cond, attn_image_indices