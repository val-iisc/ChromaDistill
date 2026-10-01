<div align="center">

<h1>ChromaDistill : Colorizing Monochrome Radiance Fields with Knowledge
Distillation</h1>
<h1>WACV 2025</h1>

<p align="center">
    <a href="https://ankitdhim.github.io/" target="_blank"><strong>Ankit Dhiman</strong></a> <sup>1,2<b></b></sup>
    ·
    <a href="https://rsrinath14.github.io/" target="_blank"><strong>R Srinath</strong></a> <sup>1<b></b></sup>
    ·
    <a href="https://www.linkedin.com/in/srinjay-sarkar-1501b9112/" target="_blank"><strong>Srinjay Sarkar</strong></a> <sup>1</sup>
    ·
    <a href="https://in.linkedin.com/in/lokesh-boregowda-14321810" target="_blank"><strong>Lokesh R Boregowda</strong></a> <sup>2</sup>
    ·
    <a href="https://cds.iisc.ac.in/faculty/venky/" target="_blank"><strong>R Venkatesh Babu</strong></a> <sup>1</sup>
</p>
<p align="center" style="padding-top: 0px;">
    <br>
    <sup>1</sup> Vision and AI Lab, IISc Bangalore
    <br>
    <sup>2</sup> Samsung R & D Institute India - Bangalore
    <br>
</p>

<a href="https://openaccess.thecvf.com/content/WACV2025/papers/Dhiman_ChromaDistill__Colorizing_Monochrome_Radiance_Fields_with_Knowledge_Distillation_WACV_2025_paper.pdf" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/WACV-ChromaDistill-blue" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2309.07668">
<img src='https://img.shields.io/badge/arxiv-ChromaDistill-red' alt='Paper PDF'></a>
<a href="https://val.cds.iisc.ac.in/chroma-distill.github.io/">
<img src='https://img.shields.io/badge/Project-Website-green' alt='Project Page'></a>
<br>
<img src='assets/Teaser-Image.jpg' alt='Teaser Image'  height='100%' width='100%'>

</div>

## Quick start

### Install environment
```bash
. ./create_env.sh
```
### Download data
```bash
. ./download_data.sh
```

### Data layout

Each scene holds the monochrome inputs and the images colorized by the teacher
network. The teacher images are looked up either in a `teacher_images` folder,
or in an `<images>_bigcolor` folder beside the inputs at the matching
resolution, which is how the released LLFF data ships them:

```
<scene>/
    poses_bounds.npy
    images/                 # monochrome inputs (1 or 3 equal channels)
    images_4/               # pre-scaled inputs, used by default
    images_4_bigcolor/      # teacher (colorized) images for images_4
```

### Colorize a scene
```bash
cd opt && . ./try_llff.sh <scene> <path_to_llff_data>
```

This runs both stages: `opt.py` fits a radiance field to the monochrome inputs,
then `opt_style.py` distils colour from the teacher images into it with the
geometry held fixed. `render_imgs.py` writes out the novel views and metrics.

## 📖 Abstract

Neural radiance field (NeRF) and Gaussian-Splatting based methods enable high-quality novel-view synthesis for multi-view images. The question arises: Can these representations generate colorized novel views when provided with monochromatic(grey-scale) inputs? Beyond its visual aesthetic significance in portraying worlds, color plays a pivotal role in downstream applications such as open-set scene decomposition. This research presents a method for synthesizing colorized novel views from input grey-scale multi-view images. Applying image or video-based colorization techniques to generated grey-scale novel views demonstrates artifacts arising from inconsistencies across views. Even training a radiance field network on colorized grey-scale image sequences fails to resolve the 3D consistency issue. We propose a distillation-based approach, leveraging knowledge from colorization networks trained on natural images and transferring it to the chosen 3D representation. Specifically, our method uses the radiance field network as a 3D representation and transfers knowledge from existing 2D colorization methods. This strategy introduces no additional weights or computational overhead to the original representation during inference. The experimental results demonstrate the superiority of our proposed method in generating high-quality colorized novel views for indoor and outdoor scenes, showcasing notable advantages in cross-view consistency compared to baseline approaches. Additionally, we illustrate the seamless extension of our method to a Gaussian-Splatting representation.

## 🤝🏼 Cite Us

```
@InProceedings{Dhiman_2025_WACV,
    author    = {Dhiman, Ankit and R, Srinath and Sarkar, Srinjay and Boregowda, Lokesh and Radhakrishnan, Venkatesh Babu},
    title     = {ChromaDistill : Colorizing Monochrome Radiance Fields with Knowledge Distillation},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {2400-2410}
}
```

## Acknowledgement:
We would like to thank [ARF](https://github.com/Kai-46/ARF-svox2) authors for open-sourcing their implementations.

