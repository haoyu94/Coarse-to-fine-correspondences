# CoFiNet: Reliable Coarse-to-fine Correspondences for Robust Point Cloud Registration (NeurIPS 2021)

PyTorch implementation of the paper:

[CoFiNet: Reliable Coarse-to-fine Correspondences for Robust Point Cloud Registration](https://arxiv.org/abs/2110.14076) by:

[Hao Yu](https://scholar.google.com/citations?hl=en&user=g7JfRn4AAAAJ), [Fu Li](https://scholar.google.com/citations?user=9a33PdMAAAAJ&hl=en), [Mahdi Saleh](https://scholar.google.com/citations?user=52yLUy0AAAAJ&hl=en), [Benjamin Busam](https://scholar.google.com/citations?user=u4rJZwUAAAAJ&hl=en) and [Slobodan Ilic](https://scholar.google.com/citations?user=ELOVd8sAAAAJ&hl=en&oi=ao).

## Introduction

We study the problem of extracting correspondences between a pair of point clouds for registration. For correspondence retrieval, existing works benefit from matching sparse keypoints detected from dense points but usually struggle to guarantee their repeatability. To address this issue, we present CoFiNet - **Co**arse-to-**Fi**ne **Net**work which extracts hierarchical correspondences from coarse to fine without keypoint detection. On a coarse scale and guided by a weighting scheme, our model firstly learns to match down-sampled nodes whose vicinity points share more overlap, which significantly shrinks the search space of a consecutive stage. On a finer scale, node proposals are consecutively expanded to patches that consist of groups of points together with associated descriptors. Point correspondences are then refined from the overlap areas of corresponding patches, by a density-adaptive matching module capable to deal with varying point density. Extensive evaluation of CoFiNet on both indoor and outdoor standard benchmarks shows our superiority over existing methods. Especially on 3DLoMatch where point clouds share less overlap, CoFiNet significantly outperforms state-of-the-art approaches by at least 5% on *Registration Recall*, with at most two-third of their parameters.

![image](https://github.com/haoyu94/Coarse-to-fine-correspondences/blob/main/figures/pipeline.jpg)

## News

+ 28.10.2021: Paper available on [arxiv](https://arxiv.org/abs/2110.14076).

+ 27.10.2021: Release training and testing code of 3DMatch and 3DLoMatch.

## Installation

+ Clone the repository:

  ```
  git clone https://github.com/haoyu94/Coarse-to-fine-correspondences.git
  cd Coarse-to-fine-correspondences
  ```
+ Create conda environment and install requirements:

  ```
  conda create -n {environment name} python=3.8
  pip install -r requirements.txt
  ```
+ Compile C++ and CUDA scripts:

  ```
  cd cpp_wrappers
  sh compile_wrappers.sh
  cd ..
  ```
  
## Demo

**TBD**

## 3DMatch & 3DLoMatch

### Pretrained model

   Pretrained model is given in `weights/`. 
   
### Prepare datasets

  ```
  sh scripts/download_data.sh
  ```
  
### Train

  ```
  sh scripts/train_3dmatch.sh
  ```
  
### Test

  + Point correspondences are first extracted by running:
  
  ```
  sh scripts/test_3dmatch.sh
  ```
  
  and stored on `snapshot/tdmatch_enc_dec_test/3DMatch/`. 
  
  
  + To evaluate on 3DLoMatch, please change the `benchmark` keyword in `configs/tdmatch/tdmatch_test.yaml` from `3DMatch` to  `3DLoMatch`.
  
  + The evaluation of extracted correspondences and relative poses estimated by RANSAC can be done by running:

  ```
  sh scripts/run_ransac.sh
  ```
  
  + The final results are stored in `est_traj/3DMatch/{number of correspondences}/result` and the results evaluated on our computer have been provided in `est_traj/`. 
  
  + To evaluate on 3DLoMatch, please change `3DMatch` in `scripts/run_ransac.sh` to `3DLoMatch`. 
 
 ## KITTI

 **TBD**
 
 ## Acknowledgments

 + The code is heavily borrowed from [PREDATOR](https://github.com/overlappredator/OverlapPredator). 
 
 + Our backbone network is from [KPConv](https://github.com/HuguesTHOMAS/KPConv).
 
 + We use the Transformer implementation in [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork). 
 
 + Sinkhorn implementation is from [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) and [RPM-Net](https://github.com/yewzijian/RPMNet).
 
 We thank the authors for their excellent work!
 
 ## Citiation
 
If you find this repository helpful, please cite:

```
@misc{yu2021cofinet,
      title={CoFiNet: Reliable Coarse-to-fine Correspondences for Robust Point Cloud Registration}, 
      author={Hao Yu and Fu Li and Mahdi Saleh and Benjamin Busam and Slobodan Ilic},
      year={2021},
      eprint={2110.14076},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
  
