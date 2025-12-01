# test_image_to_3d_pipeline.py
from typing import *
import torch
from contextlib import contextmanager
from PIL import Image

from .trellis_image_to_3d import TrellisImageTo3DPipeline
from ..modules import sparse as sp
from itertools import accumulate

from .fuse3d_utils.mask_utils import (
    fix_mask_by_majority_tensor,
    downsample_coords,
    get_valid_patches_from_images,
    exclude_patches_by_voxel_mask_attn,
)
from .fuse3d_utils.attn_utils import (
    LocalAttentionState,
    build_local_attention_hook,
    drop_attention_hook,
    cross_attention_hook_context,
)


class Fuse3DPipeline(TrellisImageTo3DPipeline):
    """
    A pipeline for testing the original SLat generation method.
    """

    def __init__(
        self,
        models=None,
        sparse_structure_sampler=None,
        slat_sampler=None,
        slat_normalization=None,
        image_cond_model=None,
    ):
        super().__init__(
            models,
            sparse_structure_sampler,
            slat_sampler,
            slat_normalization,
            image_cond_model,
        )
        self.hook_handle_list = []

    @staticmethod
    def from_pretrained(path: str) -> "Fuse3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(Fuse3DPipeline, Fuse3DPipeline).from_pretrained(
            path
        )
        new_pipeline = Fuse3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        return new_pipeline

    
    @contextmanager
    def inject_cross_attention_hooks(
        self,
        flow_model,
        attention_hook_fn,
        drop_hook_fn,
        get_attention: bool = True,
    ):
        with cross_attention_hook_context(
            flow_model=flow_model,
            attention_hook_fn=attention_hook_fn,
            drop_hook_fn=drop_hook_fn,
            get_attention=get_attention,
        ):
            yield

    def sample_slat_attention_inpaint(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
        get_attention: bool = True,
        noise: Optional[torch.Tensor] = None,
        keep_voxel_mask=None,
        gt_slat=None,
        **kwargs,
    ) -> Union[sp.SparseTensor, Tuple[sp.SparseTensor, dict]]:
        attn_state = LocalAttentionState(interval=24)

        flow_model = self.models["slat_flow_model"]
        with self.inject_cross_attention_hooks(
            flow_model,
            attention_hook_fn=build_local_attention_hook(attn_state),
            drop_hook_fn=drop_attention_hook,
            get_attention=get_attention,
        ):
            if noise is None:
                noise = sp.SparseTensor(
                    feats=torch.randn(coords.shape[0], flow_model.in_channels).to(
                        self.device
                    ),
                    coords=coords,
                )

            sampler_params = {**self.slat_sampler_params, **sampler_params}
            slat = self.slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=True,
                keep_mask=keep_voxel_mask,
                gt_slat=gt_slat,
                **kwargs,
            ).samples

            std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
            mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
            slat = slat * std + mean

            if keep_voxel_mask is not None and gt_slat is not None:
                slat.feats[keep_voxel_mask] = gt_slat.feats[keep_voxel_mask]

            if get_attention:
                attn_state.finalize()
                local_attn_state = {
                    "maps": attn_state.maps,
                    "voxel_maps": attn_state.voxel_maps,
                    "current_mean": attn_state.current_mean,
                    "current_voxel_mean": attn_state.current_voxel_mean,
                    "count": attn_state.count,
                    "interval": attn_state.interval,
                }
                return slat, local_attn_state

        return slat

    def _compute_single_image_attention(
        self,
        cond: dict,
        coords: torch.Tensor,
        heads: List[int],
        patch_index: List[int],
        valid_tokens: List[int],
        additional_token: int,
        Dcoords: torch.Tensor,
        slat_sampler_params: dict,
        local_enhancement: float,
        use_ada_lamda: bool,
    ):
        assert patch_index is not None

        if use_ada_lamda:
            enhancement = len(valid_tokens) / len(patch_index)
            enhancement = (enhancement - 1) * local_enhancement + 1
        else:
            enhancement = local_enhancement + 1

        valid_tokens = [x + additional_token for x in valid_tokens]
        patch_index = [x + additional_token for x in patch_index]

        unselected_patch_index = [
            valid_tokens.index(x) for x in valid_tokens if x not in patch_index
        ]
        patch_index = [
            valid_tokens.index(x) for x in patch_index if x in valid_tokens
        ]
        unselected_patch_index = [
            x + additional_token for x in unselected_patch_index
        ]
        patch_index = [x + additional_token for x in patch_index]

        fitted_index = list(range(additional_token)) + patch_index
        fitted_valid_index = list(range(additional_token)) + valid_tokens

        cond["cond"] = cond["cond"][:, fitted_valid_index]
        cond["neg_cond"] = cond["neg_cond"][:, fitted_valid_index]

        _, attn_dict = self.sample_slat_attention_inpaint(
            cond, coords, slat_sampler_params
        )
        attn_map = torch.stack(attn_dict["maps"], dim=0).mean(0)
        attn_map = attn_map[heads][:, :, patch_index].mean(0).sum(1)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        attn_voxel_indices = torch.nonzero(attn_map > 0.55).squeeze(1)
        attn_voxel_indices = fix_mask_by_majority_tensor(
            Dcoords[:, 1:], attn_voxel_indices
        )

        cond["cond"] = cond["cond"][:, fitted_index]
        cond["neg_cond"] = cond["neg_cond"][:, fitted_index]

        return cond, attn_voxel_indices, enhancement

    def get_attention_mask(
        self,
        coords,
        original_image: Image.Image,
        images: List[Image.Image],
        selected_patch_index_list: List,
        ori_attn_map,
        local_enhancement,
        slat_sampler_params: dict = {},
        heads: List = [0, 4, 12],
        use_ada_lamda: bool = False,
    ):
        valid_tokens_list = get_valid_patches_from_images(images)
        cond_list = [self.get_cond([image]) for image in images]
        ori_cond = self.get_cond([original_image])
        additional_token = self.models["image_cond_model"].num_register_tokens + 1

        Dcoords, downsample_idx = downsample_coords(
            coords, (2, 2, 2), require_upsample=True
        )

        attn_voxel_indices_list = []
        attn_weight_list = []

        # TODO: can be done in parallel
        for cond, patch_index, valid_tokens in zip(
            cond_list, selected_patch_index_list, valid_tokens_list
        ):
            cond, attn_voxel_indices, enhancement = self._compute_single_image_attention(
                cond=cond,
                coords=coords,
                heads=heads,
                patch_index=patch_index,
                valid_tokens=valid_tokens,
                additional_token=additional_token,
                Dcoords=Dcoords,
                slat_sampler_params=slat_sampler_params,
                local_enhancement=local_enhancement,
                use_ada_lamda=use_ada_lamda,
            )
            attn_voxel_indices_list.append(attn_voxel_indices)
            attn_weight_list.append(enhancement)

        token_len_list = [cond_list[i]["cond"].shape[1] for i in range(len(images))]
        token_cumlen_list = list(accumulate(token_len_list))
        token_cumlen_list = [0] + token_cumlen_list
        attn_weight = (
            torch.ones(
                Dcoords.shape[0], sum(token_len_list) + ori_cond["cond"].shape[1]
            )
            .to(coords.device)
            .half()
        )
        visit_record = (
            torch.zeros(
                Dcoords.shape[0],
                sum(token_len_list) + ori_cond["cond"].shape[1],
                dtype=torch.bool,
            )
            .to(coords.device)
        )

        for i, attn_indice in enumerate(attn_voxel_indices_list):
            attn_weight[
                attn_indice, token_cumlen_list[i] : token_cumlen_list[i + 1]
            ] *= attn_weight_list[i]
            visit_record[
                attn_indice, token_cumlen_list[i] : token_cumlen_list[i + 1]
            ] = True

        keep_voxel_index = torch.nonzero(
            (visit_record.sum(1) == 0)[downsample_idx]
        ).squeeze(1)
        Dsample_keep_voxel_index = torch.nonzero(
            (visit_record.sum(1) == 0)
        ).squeeze(1)

        ori_cond, attn_image_indices = exclude_patches_by_voxel_mask_attn(
            reference_image=original_image,
            attn_map=ori_attn_map,
            keep_voxel_index=Dsample_keep_voxel_index,
            ori_cond=ori_cond,
            additional_tokens_num=5,
            heads=heads,
        )

        attn_weight = attn_weight[
            :, : sum(token_len_list) + ori_cond["cond"].shape[1]
        ]
        attn_weight[
            Dsample_keep_voxel_index, sum(token_len_list) :
        ] *= (local_enhancement + 1)

        merged_cond: dict = {}
        merged_cond["cond"] = torch.cat(
            [cond_list[i]["cond"] for i in range(len(images))] + [ori_cond["cond"]],
            dim=1,
        )
        merged_cond["neg_cond"] = torch.cat(
            [cond_list[i]["neg_cond"] for i in range(len(images))]
            + [ori_cond["neg_cond"]],
            dim=1,
        )
        return merged_cond, attn_weight, keep_voxel_index, attn_image_indices

    @torch.no_grad()
    def fuse3d_run(
        self,
        reference_img: Image.Image,
        images: List[Image.Image],
        selected_patch_index_list: List,
        local_enhancement: float,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
        use_ada_lamda: bool = False,
    ):
        torch.manual_seed(seed)
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
            reference_img = self.preprocess_image(reference_img)

        reference_cond = self.get_cond([reference_img])
        coords = self.sample_sparse_structure(
            reference_cond, sampler_params=sparse_structure_sampler_params
        )
        gt_slat, ori_attn_dict = self.sample_slat_attention_inpaint(
            reference_cond, coords, slat_sampler_params
        )
        ori_attn_map = torch.stack(ori_attn_dict["voxel_maps"], dim=0).mean(0)

        merged_cond, attn_weight, keep_voxel_index, attn_image_indices = (
            self.get_attention_mask(
                coords,
                reference_img,
                images,
                selected_patch_index_list,
                ori_attn_map,
                local_enhancement,
                slat_sampler_params,
                use_ada_lamda=use_ada_lamda,
            )
        )

        slat = self.sample_slat_attention_inpaint(
            merged_cond,
            coords,
            slat_sampler_params,
            get_attention=False,
            attn_weight=attn_weight,
        )

        return (
            self.decode_slat(slat, formats),
            slat,
            keep_voxel_index,
        )