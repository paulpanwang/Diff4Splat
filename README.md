# [CVPR 2026] 🌀Diff4Splat

<h2 align="center"> <a href="https://paulpanwang.github.io/Diff4Splat">Diff4Splat: Controllable 4D Scene Generation with <br> Latent Dynamic Reconstruction Models</a></h2>

<h4 align="center">

[Panwang Pan<sup>†</sup>](https://paulpanwang.github.io), [Chenguo Lin<sup>†</sup>](https://chenguolin.github.io), [Jingjing Zhao](), [Chenxin Li](), [Yuchen Lin](https://wgsxm.github.io), [Haopeng Li](), [Honglei Yan](https://openreview.net/profile?id=~Honglei_Yan1), [Kairun Wen](), [Yunlong Lin](), [Yixuan Yuan](), [Yadong Mu](http://www.muyadong.com)

[![arXiv](https://img.shields.io/badge/arXiv-2511.00503-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.00503)
[![Project Page](https://img.shields.io/badge/🏠-Project%20Page-blue.svg)](https://paulpanwang.github.io/Diff4Splat/)
[![YouTube-Video](https://img.shields.io/badge/YouTube-Video-red.svg?logo=youtube)](https://paulpanwang.github.io/Diff4Splat/)
[![Model](https://img.shields.io/badge/🤗%20Model-Diff4Splat-yellow.svg)](https://huggingface.co/paulpanwang/Diff4Splat)
[![Demo](https://img.shields.io/badge/🤗%20Demo-Diff4Splat-green.svg)](https://huggingface.co/paulpanwang/Diff4Splat)

<p align="center">
    <img width="90%" alt="teaser" src="./assets/teaser.png">
</p>

</h4>


Diff4Splat is a feed-forward method that synthesizes controllable and explicit 4D scenes from a single image. Our approach unifies the generative priors of video diffusion models with geometry and motion constraints learned from large-scale 4D datasets.


Here is our [Project Page](https://paulpanwang.github.io/Diff4Splat/).

Feel free to contact us or open an issue if you have any questions or suggestions.

## 🔥 See Also
You may also be interested in our other works:
- [**[CVPR 2026] MoVieS**](https://github.com/chenguolin/MoVieS): a feed-forward model for 4D dynamic reconstruction from monocular videos.

## 📢 News
- **2026-02-21**: The paper is accepted to CVPR 2026.
- **2025-11-01**: Diff4Splat is released on arXiv.
- **2025-10-15**: Initial codebase structure established.
- **2025-10-01**: Project development started.



## 📝 Paper vs Released Code

This section clarifies the differences between the paper description and the current released codebase, to help set expectations for reproducibility.

### What the Paper Describes
- **Video Backbone**: CogVideoX-style Video DiT with 32-channel 3D Causal VAE (4×8×8 compression)
- **LDRM Input**: Video latent tensor (z) from the diffusion model, together with camera information, processed by LDRM Transformer to output deformable 3D Gaussians



### Release Roadmap
We are actively working on:
1. **Paper-faithful implementation**: A version closer to the CogVideoX + latent-input LDRM stack described in the paper
2. **Complete training/inference scripts**: Exact scripts to reproduce the paper's results
3. **Pretrained checkpoints**: Both the paper's setup and this repository's engineering variant


## 📋 Project Status
- [x] Inference code released
- [x] Training code and data preprocessing scripts released
- [ ] Pretrained checkpoints (coming soon)
- [ ] HuggingFace demo (coming soon)
- [ ] Preprocessed dataset (coming soon)

## 🔧 Installation

### Requirements
- Python >= 3.10
- PyTorch >= 2.0 (with CUDA support)
- CUDA >= 11.8

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/paulpanwang/Diff4Splat.git
cd Diff4Splat

# Install required packages
pip install -r settings/requirements.txt
```

## 📊 Datasets

Configure your dataset root path in `src/options.py` or via the `DATASET_ROOT` environment variable.

The following datasets are supported:
- **RealEstate10K** (`re10k`) - Static scenes
- **TartanAir** (`tartanair`) - Static scenes
- **MatrixCity** (`matrixcity`) - Static scenes
- **DL3DV** (`dl3dv`) - Static scenes
- **DynamicReplica** (`dynamicreplica`) - Dynamic scenes
- **PointOdyssey** (`pointodyssey`) - Dynamic scenes
- **VKITTI2** (`vkitti2`) - Dynamic scenes
- **Spring** (`spring`) - Dynamic scenes
- **Stereo4D** (`stereo4d`) - Dynamic scenes

Dataset paths can be configured in `src/options.py`:



### Questions & Feedback
If you have questions about reproducibility or comparisons, please open an issue or contact the authors. We appreciate your understanding as we continue to improve and complete this codebase!





## 📚 Citation

If you find our work helpful, please consider citing:

```bibtex
@article{pan2025diff4splat,
  title={Diff4Splat: Controllable 4D Scene Generation with Latent Dynamic Reconstruction Models},
  author={Pan, Panwang and Lin, Chenguo and Zhao, Jingjing and Li, Chenxin and Lin, Yuchen and Li, Haopeng and Yan, Honglei and Wen, Kairun and Lin, Yunlong and Yuan, Yixuan and Mu, Yadong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 😊 Acknowledgement

We would like to thank the authors of [MoVieS](https://github.com/chenguolin/MoVieS), [PartCrafter](https://github.com/wgsxm/PartCrafter), [DiffSplat](https://chenguolin.github.io/projects/DiffSplat), and other related works for their inspiring research and open-source contributions that helped shape this project.


