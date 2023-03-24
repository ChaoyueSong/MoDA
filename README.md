<!-- # MoDA: Modeling Deformable 3D Objects from Casual Videos
#### [Project](https://chaoyuesong.github.io/MoDA/) |   [Paper](https://openreview.net/pdf?id=fG01Z_unHC)

**MoDA: Modeling Deformable 3D Objects from Casual Videos** <br>
[Chaoyue Song](https://chaoyuesong.github.io/), Tianyi Chen, Yiwen Chen, Jiacheng Wei, [Chuan-Sheng Foo](http://ai.stanford.edu/~csfoo/), [Fayao Liu](https://sites.google.com/site/fayaoliu/),
[Guosheng Lin](https://guosheng.github.io/) <br>
in arXiv, 2023. -->

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">MoDA: Modeling Deformable 3D Objects from Casual Videos</h1>
  <p align="center">
    <a href="https://chaoyuesong.github.io"><strong>Chaoyue Song</strong></a>
    ·
    <a href=""><strong>Tianyi Chen</strong></a>
    ·
    <a href=""><strong>Yiwen Chen</strong></a>
    ·
    <a href=""><strong>Jiacheng Wei</strong></a>
      ·
    <a href="http://ai.stanford.edu/~csfoo/"><strong>Chuan-Sheng Foo</strong></a>
      ·
    <a href="https://sites.google.com/site/fayaoliu/"><strong>Fayao Liu</strong></a>
      ·
    <a href="https://guosheng.github.io/"><strong>Guosheng Lin</strong></a>
  </p>
  <h2 align="center">arXiv 2023</h2>
  <div align="center">
    <img src="./assets/teaser.gif" alt="Logo" width="100%">
  </div>

  <p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
    <a href='https://colab.research.google.com/drive/1-AWeWhPvCTBX0KfMtgtMk10uPU05ihoA?usp=sharing' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a>
    <a href="https://huggingface.co/spaces/Yuliang/ICON"  style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-orange'></a><br></br>
    <a href='https://arxiv.org/abs/2112.09127'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <a href='https://icon.is.tue.mpg.de/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/ICON-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=orange' alt='Project Page'>
    <a href="https://discord.gg/Vqa7KBGRyk"><img src="https://img.shields.io/discord/940240966844035082?color=7289DA&labelColor=4a64bd&logo=discord&logoColor=white&style=for-the-badge"></a>
    <a href="https://youtu.be/hZd6AYin2DE"><img alt="youtube views" title="Subscribe to my YouTube channel" src="https://img.shields.io/youtube/views/hZd6AYin2DE?logo=youtube&labelColor=ce4630&style=for-the-badge"/></a>
  </p>
</p>

<br />
<br />

## News :triangular_flag_on_post:

- [2022/12/15] ICON belongs to the past, [ECON](https://github.com/YuliangXiu/ECON) is the future!
- [2022/09/12] Apply [KeypointNeRF](https://markomih.github.io/KeypointNeRF/) on ICON, quantitative numbers in [evaluation](docs/evaluation.md#benchmark-train-on-thuman20-test-on-cape)
- [2022/07/30] <a href="https://huggingface.co/spaces/Yuliang/ICON"  style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-orange'></a> <a href='https://colab.research.google.com/drive/1-AWeWhPvCTBX0KfMtgtMk10uPU05ihoA?usp=sharing' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a> are both available
- [2022/07/26] New cloth-refinement module is released, try `-loop_cloth`
- [2022/06/13] ETH Zürich students from 3DV course create an add-on for [garment-extraction](docs/garment-extraction.md)
- [2022/05/16] <a href="https://github.com/Arthur151/ROMP">BEV</a> is supported as optional HPS by <a href="https://scholar.google.com/citations?hl=en&user=fkGxgrsAAAAJ">Yu Sun</a>, see [commit #060e265](https://github.com/YuliangXiu/ICON/commit/060e265bd253c6a34e65c9d0a5288c6d7ffaf68e)
- [2022/05/15] Training code is released, please check [Training Instruction](docs/training.md)
- [2022/04/26] <a href="https://github.com/Jeff-sjtu/HybrIK">HybrIK (SMPL)</a> is supported as optional HPS by <a href="https://jeffli.site/">Jiefeng Li</a>, see [commit #3663704](https://github.com/YuliangXiu/ICON/commit/36637046dcbb5667cdfbee3b9c91b934d4c5dd05)
- [2022/03/05] <a href="https://github.com/YadiraF/PIXIE">PIXIE (SMPL-X)</a>, <a href="https://github.com/mkocabas/PARE">PARE (SMPL)</a>, <a href="https://github.com/HongwenZhang/PyMAF">PyMAF (SMPL)</a> are all supported as optional HPS

<br>
