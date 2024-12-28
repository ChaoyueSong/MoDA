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

   ### [Project](https://chaoyuesong.github.io/MoDA/) | [Paper](https://chaoyuesong.github.io/MoDA/MoDA.pdf) | [Video](https://www.youtube.com/watch?v=6RAPy8DLv-E)
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

## Training
#### 1. cat-pikachiu (casual-cat) 

```bash
# We store images as lines of pixels following BANMo. 
# only needs to run it once per sequence and data are stored in
# database/DAVIS/Pixel
python preprocess/img2lines.py --seqname cat-pikachiu

# Training
bash scripts/template.sh 0,1 cat-pikachiu 10001 "no" "no"
# argv[1]: gpu ids separated by comma 
# args[2]: sequence name
# args[3]: port for distributed training
# args[4]: use_human, pass "" for human, "no" for others
# args[5]: use_symm, pass "" to force x-symmetric shape

# Extract articulated meshes and render
bash scripts/render_mgpu.sh 0 cat-pikachiu logdir/cat-pikachiu-e120-b256-ft2/params_latest.pth \
        "0 1 2 3 4 5 6 7 8 9 10" 256
# argv[1]: gpu id
# argv[2]: sequence name
# argv[3]: weights path
# argv[4]: video id separated by space
# argv[5]: resolution of running marching cubes (256 by default)
```

#### 2. adult7 (casual-human) 

```bash
python preprocess/img2lines.py --seqname adult7
bash scripts/template.sh 0,1 adult7 10001 "" ""
bash scripts/render_mgpu.sh 0 adult7 logdir/adult7-e120-b256-ft2/params_latest.pth \
        "0 1 2 3 4 5 6 7 8 9" 256
```

### TODO
- [x] Inital code release.
- [ ] Code cleaning and further checking.
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