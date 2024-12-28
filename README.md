<div align="center">

  <h1 align="center">MoDA: Modeling Deformable 3D Objects from Casual Videos</h1>
  <div>
    <a href="https://chaoyuesong.github.io"><strong>Chaoyue Song</strong></a>
    ·
    <a href="https://plusmultiply.github.io/"><strong>Jiacheng Wei</strong></a>
      ·
    <a href="https://bravotty.github.io/"><strong>Tianyi Chen</strong></a>
    ·
    <a href="https://buaacyw.github.io/"><strong>Yiwen Chen</strong></a>
    ·
    <a href="http://ai.stanford.edu/~csfoo/"><strong>Chuan-Sheng Foo</strong></a>
      ·
    <a href="https://sites.google.com/site/fayaoliu/"><strong>Fayao Liu</strong></a>
      ·
    <a href="https://guosheng.github.io/"><strong>Guosheng Lin</strong></a>
  </div>
  
   ### IJCV 2024

   ### [Project](https://chaoyuesong.github.io/MoDA/) | [Paper](https://chaoyuesong.github.io/MoDA/MoDA.pdf)
<tr>
    <img src="https://github.com/ChaoyueSong/ChaoyueSong.github.io/blob/gh-pages/files/project/moda/teaser.gif" width="70%"/>
</tr>
</div>
<br />






## Installation

We test our method on torch 1.10 + cu113

```bash
# clone repo
git clone https://github.com/ChaoyueSong/MoDA.git --recursive
cd MoDA
# create conda env
conda env create -f misc/moda.yml
conda activate moda
# install pytorch3d, kmeans-pytorch
pip install -e third_party/pytorch3d
pip install -e third_party/kmeans_pytorch
# install detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

## Data preparation

For casual-human (adult7) and casual-cat (cat-pikachiu) used in this work, you can download the pre-processed data as in BANMo, plz check the license for these data in [BANMo](https://github.com/facebookresearch/banmo/).
```bash
# (~8G for each)
bash misc/processed/download.sh cat-pikachiu
bash misc/processed/download.sh human-cap
```
For AMA and Synthetic data, please check [here](https://github.com/facebookresearch/banmo/tree/main/scripts).
</details>


**To use your own videos, or pre-process raw videos into our format, 
please follow this [instruction](https://github.com/facebookresearch/banmo/tree/main/preprocess).**

## PoseNet weights

Download pre-trained PoseNet weights for human and quadrupeds.
```bash
mkdir -p mesh_material/posenet && cd "$_"
wget $(cat ../../misc/posenet.txt); cd ../../
```

### TODO
- [x] Release the dataset and data preprocess codes.
- [ ] Release training code.
- [ ] Release the pretrained models.


## Citation

```bibtex
@article{song2024moda,
  title={Moda: Modeling deformable 3d objects from casual videos},
  author={Song, Chaoyue and Wei, Jiacheng and Chen, Tianyi and Chen, Yiwen and Foo, Chuan-Sheng and Liu, Fayao and Lin, Guosheng},
  journal={International Journal of Computer Vision},
  pages={1--20},
  year={2024},
  publisher={Springer}
}
```

<br/>

## Acknowledgments

We thank [BANMo](https://github.com/facebookresearch/banmo) for their code and data.