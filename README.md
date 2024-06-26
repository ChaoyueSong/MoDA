<div align="center">

  <h1 align="center">MoDA: Modeling Deformable 3D Objects from Casual Videos</h1>
  <div>
    <a href="https://chaoyuesong.github.io"><strong>Chaoyue Song</strong></a>
     ·
    <a href="https://plusmultiply.github.io/"><strong>Jiacheng Wei</strong></a>
    ·
    <a href="https://bravotty.github.io/"><strong>Tianyi Chen</strong></a>
    ·
    Yiwen Chen
      ·
    <a href="http://ai.stanford.edu/~csfoo/"><strong>Chuan-Sheng Foo</strong></a>
      ·
    <a href="https://sites.google.com/site/fayaoliu/"><strong>Fayao Liu</strong></a>
      ·
    <a href="https://guosheng.github.io/"><strong>Guosheng Lin</strong></a>
  </div>
  
   ### arXiv 2023

   ### [Project](https://chaoyuesong.github.io/MoDA/) | [Paper](https://chaoyuesong.github.io/MoDA/MoDA.pdf)
<tr>
    <img src="https://github.com/ChaoyueSong/ChaoyueSong.github.io/blob/gh-pages/files/project/moda/teaser.gif" width="70%"/>
</tr>
</div>
<br />

## Highlights :star2:

- We propose neural dual quaternion blend skinning (NeuDBS) as our deformation model to replace LBS, which can resolve the skin-collapsing artifacts.
- Introduce a texture filtering approach for texture rendering that effectively minimizes the impact of noisy colors outside target deformable objects.
- Formulate the 2D-3D matching as an optimal transport problem that helps to refine the bad segmentation obtained from a off-the-shelf method and predict the consistent 3D shape.

<br>

## News :triangular_flag_on_post:

- [2023/04/18] Our paper is available on [arXiv](http://arxiv.org/abs/2304.08279) now.

<br>

## Reconstruction results
We compare reconstruction results of MoDA and BANMo, the skin-collapsing artifacts of BANMo are marked with red circles. Please refer to our [Project](https://chaoyuesong.github.io/MoDA/) page for more reconstruction results.

https://user-images.githubusercontent.com/56154447/227527982-43b25d28-34a5-4b5a-9254-eaf5492e9d80.mp4

<br>
BANMo has more obvious skin-collapsing artifacts for motion with large rotations, our method can resolves the artifacts with the proposed NeuDBS.

<tr>
    <img src="https://chaoyuesong.github.io/MoDA/imgs/deformation_sequence.png" width="70%"/>
</tr>

## 2D-3D matching via optimal transport
By registering 2D pixels across different frames with optimal transport, we can refine the bad segmentation and predict the consistent 3D shape of the cat.
<tr>
    <img src="https://chaoyuesong.github.io/MoDA/imgs/moda_ot_demo.jpg" width="70%"/>
</tr>

## Texture filtering
We show the effectiveness of texture filtering appraoch by adding it to both MoDA and BANMo.


https://user-images.githubusercontent.com/56154447/232507580-ccdf9170-76c0-49a5-b21b-ca2b11d19c04.mp4


## Application: motion re-targeting
We compare the motion re-targeting results of MoDA and BANMo.

https://user-images.githubusercontent.com/56154447/227528406-e883c13a-88cf-40d3-a1a9-f00dbf25214e.mp4

## Citation

```bibtex
@article{song2023moda,
  title={MoDA: Modeling Deformable 3D Objects from Casual Videos},
  author={Song, Chaoyue and Chen, Tianyi and Chen, Yiwen and Wei, Jiacheng and Foo, Chuan Sheng and Liu, Fayao and Lin, Guosheng},
  journal={arXiv preprint arXiv:2304.08279},
  year={2023}
}
```

<br/>

## Acknowledgments

We thank [BANMo](https://github.com/facebookresearch/banmo) for their code and data.
