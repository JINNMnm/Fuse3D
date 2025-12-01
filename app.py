import gradio as gr
from gradio_litmodel3d import LitModel3D
import sys
from typing import *
import shutil
import numpy as np
import torch
from PIL import Image
import trimesh
import os
from easydict import EasyDict as edict
from pathlib import Path
import imageio

sys.path.append(".")
from fuse3d.representations import Gaussian, MeshExtractResult
from fuse3d.pipelines import Fuse3DPipeline
from fuse3d.utils import render_utils, postprocessing_utils

MAX_SEED = np.iinfo(np.int32).max
MAX_EDITORS = 5
TMP_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "app_outputs"
EXAMPLE_ROOT = Path("./assets")
GLOBAL_DIR = EXAMPLE_ROOT / "global_cond"
LOCAL_COND_DIR = EXAMPLE_ROOT / "local_cond"


def prepare_dynamic_examples() -> List[List[Any]]:
    example_data: List[List[Any]] = []

    if not GLOBAL_DIR.exists():
        return []

    global_files = sorted([f for f in GLOBAL_DIR.iterdir() if f.is_file()])

    for global_file in global_files:
        cond_name = global_file.stem
        local_folder = LOCAL_COND_DIR / cond_name
        
        if not local_folder.is_dir():
            continue

        local_files = sorted([f for f in local_folder.iterdir() if f.is_file()])
        
        bg_files = sorted([f for f in local_files if f.name.lower().startswith("bg")])
        com_files = sorted([f for f in local_files if f.name.lower().startswith('com')])
        layer_files = sorted([f for f in local_files if f.name.lower().startswith('layer')])

        image_triplets: List[Tuple[Path, Path, Any]] = []
        
        for i in range(min(len(bg_files), len(com_files))):
            bg_path = bg_files[i]
            com_path = com_files[i]
            
            bg_suffix = bg_path.stem.replace('bg', '')
            com_suffix = com_path.stem.replace('com', '')
            
            if bg_suffix == com_suffix:
                layer_path = None
                for lf in layer_files:
                    if lf.stem.replace('layer', '') == bg_suffix:
                        layer_path = lf
                        break
                
                image_triplets.append((bg_path, com_path, layer_path))

        editor_inputs: List[Dict[str, Any]] = []

        for i, (bg_path, com_path, layer_path) in enumerate(image_triplets):
            if i >= MAX_EDITORS:
                break

            editor_dict = {
                "background": str(bg_path.resolve()),
                "composite": str(com_path.resolve()),
                "layers": [str(layer_path.resolve())] if layer_path else None,
            }
            editor_inputs.append(editor_dict)

        single_example = [str(global_file.resolve())] 
        
        for editor_input in editor_inputs:
            single_example.append(editor_input)

        for i in range(len(editor_inputs), MAX_EDITORS):
            single_example.append(None)

        example_data.append(single_example)

    return example_data

def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def add_editor(vis_list):
    for i, visible in enumerate(vis_list):
        if not visible:
            vis_list[i] = True
            break
    updates = [gr.update(visible=v) for v in vis_list]
    return vis_list, *updates


def delete_editor(index, vis_list):
    vis_list[index] = False
    updates = [gr.update(visible=v) for v in vis_list]
    return vis_list, *updates


def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def process_image_editor(background, composite, patch_size=14, threshold=5):
    background = np.array(background)
    composite = np.array(composite)
    if np.array_equal(background, composite):
        return None

    assert background.shape == composite.shape

    diff = np.linalg.norm(
        background.astype(np.float32) - composite.astype(np.float32), axis=-1
    )

    mask = diff > threshold

    H, W = mask.shape
    h_blocks = H // patch_size
    w_blocks = W // patch_size

    mask_cropped = mask[: h_blocks * patch_size, : w_blocks * patch_size]
    mask_patches = mask_cropped.reshape(h_blocks, patch_size, w_blocks, patch_size)
    mask_patches = np.any(mask_patches, axis=(1, 3))
    selected_patches = mask_patches.flatten()
    selected_index = np.nonzero(selected_patches)[0].tolist()
    return selected_index


def process_image_editors(visibility, *args):
    images_list = []
    selected_patches_index_list = []
    for i, vis in enumerate(visibility):
        if vis:
            images_list.append(args[i]["background"])
            selected_patches_index_list.append(
                process_image_editor(args[i]["background"], args[i]["composite"])
            )
    return images_list, selected_patches_index_list


def save_slat_selected_voxel(slat, selected_coords, req: gr.Request):
    coords = slat.coords.cpu().numpy()[:, 1:]
    return save_selected_voxel(coords, selected_coords, req)


def save_selected_voxel(coords, selected_coords, req: gr.Request):
    encoding = trimesh.voxel.encoding.SparseBinaryEncoding(coords)
    voxel_grid = trimesh.voxel.VoxelGrid(encoding)
    colors_on_voxel = np.zeros(
        (
            int(voxel_grid.shape[0]),
            int(voxel_grid.shape[1]),
            int(voxel_grid.shape[2]),
            3,
        ),
        dtype=np.float32,
    )
    colors = np.ones((len(coords), 3))
    colors[:] = [1, 0, 0]
    if selected_coords is not None:
        colors[selected_coords, :] = [0, 1, 0]
    for i, coord in enumerate(coords):
        colors_on_voxel[coord[0], coord[1], coord[2]] = colors[i]
    voxel_mesh = voxel_grid.as_boxes(colors_on_voxel)
    voxel_mesh.export(TMP_DIR / req.session_hash / f"{req.session_hash}.glb")

    return TMP_DIR / req.session_hash / f"{req.session_hash}.glb"


def preprocess_image(image) -> Image.Image:
    if isinstance(image, dict):
        image["background"] = pipeline.preprocess_image(image["background"])
        return image
    else:
        image = image
        return pipeline.preprocess_image(image)


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        "gaussian": {
            **gs.init_params,
            "_xyz": gs._xyz.cpu().numpy(),
            "_features_dc": gs._features_dc.cpu().numpy(),
            "_scaling": gs._scaling.cpu().numpy(),
            "_rotation": gs._rotation.cpu().numpy(),
            "_opacity": gs._opacity.cpu().numpy(),
        },
        "mesh": {
            "vertices": mesh.vertices.cpu().numpy(),
            "faces": mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state["gaussian"]["aabb"],
        sh_degree=state["gaussian"]["sh_degree"],
        mininum_kernel_size=state["gaussian"]["mininum_kernel_size"],
        scaling_bias=state["gaussian"]["scaling_bias"],
        opacity_bias=state["gaussian"]["opacity_bias"],
        scaling_activation=state["gaussian"]["scaling_activation"],
    )
    gs._xyz = torch.tensor(state["gaussian"]["_xyz"], device="cuda")
    gs._features_dc = torch.tensor(state["gaussian"]["_features_dc"], device="cuda")
    gs._scaling = torch.tensor(state["gaussian"]["_scaling"], device="cuda")
    gs._rotation = torch.tensor(state["gaussian"]["_rotation"], device="cuda")
    gs._opacity = torch.tensor(state["gaussian"]["_opacity"], device="cuda")

    mesh = edict(
        vertices=torch.tensor(state["mesh"]["vertices"], device="cuda"),
        faces=torch.tensor(state["mesh"]["faces"], device="cuda"),
    )

    return gs, mesh


def generate(
    reference_image,
    condition_images,
    selected_patches_list,
    local_enhancement,
    seed,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength,
    slat_sampling_steps,
    req: gr.Request,
):
    outputs, slat, keep_voxel_index = pipeline.fuse3d_run(
        reference_image,
        condition_images,
        selected_patches_list,
        local_enhancement,
        seed,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    update_mask = [i for i in range(slat.coords.shape[0]) if i not in keep_voxel_index]

    video = render_utils.render_video(
        outputs["gaussian"][0], num_frames=120, bg_color=(1, 1, 1)
    )["color"]
    video_path = TMP_DIR / req.session_hash / "sample.mp4"
    imageio.mimsave(video_path, video, fps=15)
    torch.cuda.empty_cache()
    state = pack_state(outputs["gaussian"][0], outputs["mesh"][0])
    return video_path, slat, state, update_mask


def extract_glb(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    req: gr.Request,
) -> Tuple[str, str]:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(
        gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False
    )
    glb_path = os.path.join(user_dir, "sample.glb")
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path


def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, "sample.ply")
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


with gr.Blocks() as demo:
    editor_visibility = gr.State([True] + [False] * (MAX_EDITORS - 1))
    images_conditions = gr.State(None)
    selected_patches_index_list = gr.State(None)
    SLat_prior = gr.State(None)
    output_buf = gr.State()
    update_mask = gr.State(None)
    image_editors = []

    with gr.Row():
        with gr.Column(scale=1):
            auto_reference_img = gr.Image(
                label="Global Image",
                format="png",
                image_mode="RGBA",
                type="pil",
                height=300,
            )

            with gr.Accordion("generation settings", open=False):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)
                local_enhancement = gr.Slider(
                    0, 10, label="Local Enhancement", value=0, step=1
                )
                with gr.Row():
                    ss_guidance_strength = gr.Slider(
                        0.0,
                        10.0,
                        label="Sparse SLat Guidance Strength",
                        value=7.5,
                        step=0.1,
                    )
                    ss_sampling_steps = gr.Slider(
                        1, 50, label="Sparse SLat Sampling Steps", value=12, step=1
                    )
                with gr.Row():
                    slat_guidance_strength = gr.Slider(
                        0.0, 10.0, label="SLat Guidance Strength", value=3.0, step=0.1
                    )
                    slat_sampling_steps = gr.Slider(
                        1, 50, label="SLat Sampling Steps", value=12, step=1
                    )

            auto_generate_button = gr.Button("Generate")
            with gr.Accordion(label="GLB Extraction Settings", open=False):
                mesh_simplify = gr.Slider(
                    0.9, 0.98, label="Simplify", value=0.95, step=0.01
                )
                texture_size = gr.Slider(
                    512, 2048, label="Texture Size", value=1024, step=512
                )

            with gr.Row():
                extract_glb_btn = gr.Button("Extract GLB", interactive=False)
                extract_gs_btn = gr.Button("Extract Gaussian", interactive=False)
            add_button = gr.Button("➕ Add Local Image Condition")

            editor_boxes = []
            delete_buttons = []

            gr.Markdown(
                "<small>💡 **Tip:** Use the **Pen** 🖌️ and **Eraser** 🧽 tools to manually draw masks.</small>"
            )

            for i in range(MAX_EDITORS):
                with gr.Row(visible=(i == 0)) as editor_row:
                    image_editor = gr.ImageEditor(
                        label=f"Image Condition{i+1}",
                        type="pil",
                        container=False,
                        layers=False,
                    )
                    image_editors.append(image_editor)
                    delete_btn = gr.Button("🗑️ Delete", scale=0)

                    editor_boxes.append(editor_row)
                    delete_buttons.append(delete_btn)

        with gr.Column(scale=1):
            voxel_output = LitModel3D(label="selected_voxel", exposure=10.0, height=300)
            video_output = gr.Video(
                label="sample_rf", autoplay=True, loop=True, height=300
            )
            model_output = LitModel3D(
                label="Extracted GLB/Gaussian", exposure=10.0, height=300
            )
            with gr.Row():
                download_glb = gr.DownloadButton(
                    label="Download GLB", interactive=False
                )
                download_gs = gr.DownloadButton(
                    label="Download Gaussian", interactive=False
                )
    with gr.Row() as example_row:
        gr.Examples(
            examples=prepare_dynamic_examples(),
            inputs=[
                auto_reference_img,
                *image_editors,
            ],
        )

    demo.load(start_session)
    demo.unload(end_session)

    # main generation events
    auto_generate_button.click(
        fn=get_seed, inputs=[randomize_seed, seed], outputs=[seed]
    ).then(
        fn=process_image_editors,
        inputs=[editor_visibility, *image_editors],
        outputs=[images_conditions, selected_patches_index_list],
    ).then(
        fn=generate,
        inputs=[
            auto_reference_img,
            images_conditions,
            selected_patches_index_list,
            local_enhancement,
            seed,
            ss_guidance_strength,
            ss_sampling_steps,
            slat_guidance_strength,
            slat_sampling_steps,
        ],
        outputs=[
            video_output,
            SLat_prior,
            output_buf,
            update_mask,
        ],
    ).then(
        fn=save_slat_selected_voxel,
        inputs=[SLat_prior, update_mask],
        outputs=[voxel_output],
    ).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    # preprocess events
    for image_editor in image_editors:
        image_editor.upload(
            fn=preprocess_image,
            inputs=[image_editor],
            outputs=[image_editor],
        )
    auto_reference_img.upload(
        preprocess_image,
        inputs=[auto_reference_img],
        outputs=[auto_reference_img],
    )

    # extraction events
    extract_glb_btn.click(
        extract_glb,
        inputs=[output_buf, mesh_simplify, texture_size],
        outputs=[model_output, download_glb],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_glb],
    )
    extract_gs_btn.click(
        extract_gaussian,
        inputs=[output_buf],
        outputs=[model_output, download_gs],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_gs],
    )

    # clear events
    model_output.clear(
        lambda: gr.Button(interactive=False),
        outputs=[download_glb],
    )
    video_output.clear(
        lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False)]),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    # editor events
    add_button.click(
        fn=add_editor,
        inputs=[editor_visibility],
        outputs=[editor_visibility] + editor_boxes,
    )

    for i, btn in enumerate(delete_buttons):
        btn.click(
            fn=delete_editor,
            inputs=[gr.State(i), editor_visibility],
            outputs=[editor_visibility] + editor_boxes,
        )


pipeline = Fuse3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
demo.launch()