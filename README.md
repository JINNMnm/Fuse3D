<p align="center">

  <h2 align="center">Fuse3D: Generating 3D Assets Controlled by Multi-Image Fusion</h2>
  <p align="center">
    <strong>Xuancheng Jin</strong>
    ·
    <strong>Rengan Xie</strong>
    ·
    <strong>Wenting Zheng</strong>
    ·
    <a href="https://kkbless.github.io/"><strong>Rui Wang</strong></a>
    ·
    <a href="https://rfidblog.org.uk/"><strong>Hujun Bao</strong></a>
    ·
    <a href="http://www.cad.zju.edu.cn/home/huo/"><strong>Yuchi Huo</strong></a>
    <br>
    <br>
        <a href='https://jinnmnm.github.io/Fuse3d.github.io/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
    <br>
  </p>
  
  <table align="center">
    <tr>
    <td>
      <img src="assets/teaser.png" alt='teaser1'>
    </td>
    </tr>
  </table>

## 📢 News
* **[Sept.15.2025]** Fuse3D is accepted to SIGGRAPH Asia 2025. The code is still being organized. Stay tuned for updates!

* **[Nov.30.2025]** The code is publicly released!


## ⚙️Installation

### Prerequisites
- **System**: The code is tested on **Linux**.
- **Hardware**: An NVIDIA GPU with at least 16GB of memory is necessary. The code has been verified on NVIDIA A6000 GPUs.  
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain submodules. The code has been tested with CUDA versions 11.8.  
  - Python version 3.8 or higher is required. 

### Installation Steps
1. Clone the repo:
    ```sh
    git clone https://github.com/JINNMnm/Fuse3D.git
    cd Fuse3D
    ```

2. Install the dependencies:

    As Fuse3D builds upon [TRELLIS](https://github.com/microsoft/TRELLIS). You can find more details about the dependencies in the TRELLIS repository.

    ```sh
    . ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
    ```

  ## 🚀 Pretrained Models
  We do not modify the pretrained models of TRELLIS. The weights will be automatically downloaded when you run:
  ```sh
  Fuse3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
  ```
  Optionally, you can manually download the weights from [HuggingFace](https://huggingface.co/microsoft/TRELLIS-image-large) and change the path in the above command to the local path.
  ```sh
  Fuse3DPipeline.from_pretrained("path/to/local/directory")
  ``` 

  ## 💡 Inference

  Since our method involves manual masking, we provide an interactive **Gradio demo** to facilitate the process and simplify testing.

  You can launch the web interface by running:
  ```sh
  python app.py
  ```

  ## 📜 Acknowledgements
  This code builds upon [TRELLIS](https://github.com/microsoft/TRELLIS). We sincerely thank the authors for their great work and open-sourcing the code.