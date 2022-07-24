## Injecting 3D Perception of Controllable NeRF-GAN into StyleGAN for Editable Portrait Image Synthesis

<img src="./assets/thumbnail.png" width="90%">

## [Project page](https://jgkwak95.github.io/surfgan/) | [Paper](http://arxiv.org/abs/2207.10257)  
<br>
  
>**Injecting 3D Perception of Controllable NeRF-GAN into StyleGAN for Editable Portrait Image Synthesis** <br>
>[Jeong-gi Kwak](https://jgkwak95.github.io/), Yuanming Li, Dongsik Yoon, Donghyeon Kim, David Han, Hanseok Ko<br>
>**ECCV 2022** <br>

<br>

This repository includes the official Pytorch implementation of SURF-GAN. <br>


## SURF-GAN 

SURF-GAN, which is a NeRF-based 3D-aware GAN, can discover disentangled semantic attributes in an unsupervised manner.  <br><br>
<img src="https://user-images.githubusercontent.com/67986601/180636584-6ac7f46f-ff8f-4da1-a4b9-69e31e769909.gif" width=150> <br>
(Tranined on 64x64 CelebA and rendered with 256x256) <br>

<!-- ### Envs -->
<!-- ### Training -->
<!-- ### Evaluation -->
<!-- ### Generation  -->
<!-- ### Semantic attributes discovery -->

### Get started
Instructions will be updated. <br>

## 3D-Controllable StyleGAN

Injecting the 3D prior of SURF-GAN into StyleGAN.


<img src="https://jgkwak95.github.io/surfgan/assets/3d_stylegan.png" width="50%"> <br>

### Video
<img src="https://user-images.githubusercontent.com/67986601/180636660-189a6479-084a-460f-ac69-a6491987d9b8.gif" width="50%"> <br>

<img src="https://user-images.githubusercontent.com/67986601/180636672-4b22539a-73ee-470c-b5f8-795df3a27a5c.gif" width="50%"> <br>


<img src="https://user-images.githubusercontent.com/67986601/180636702-a0b7f7da-774b-4374-a44c-3d0b49f03fd8.gif" width="50%"> <br>


### + Style
Also, it is compatible with numerous StyleGAN-based techniques, e.g., [Toonifying](https://github.com/justinpinkney/toonify). <br>
<img src="https://user-images.githubusercontent.com/67986601/180636715-39a4438e-9176-4c3e-b8bd-b417c5b6d4c5.gif" width="60%"> <br>

<img src="https://user-images.githubusercontent.com/67986601/180636724-0963eb04-5ec1-441c-8305-464c4796ef02.gif" width="60%"> <br>


### Limitation 
Our 3D controllable StyleGAN is not based on 3D representations such as mesh or NeRF, so as you can see when it comes to video generation, it shows the problem of “texture sticking” pointed out in [StyleGAN3](https://nvlabs.github.io/stylegan3/) (especially in hair and beard). That is one of the most noticable artifacts in GAN generated videos. We expect this to be mitigated with StyleGAN3. <br>
<img src="https://user-images.githubusercontent.com/67986601/180636762-dfe6a900-eaeb-4b12-8b82-1fb70c06dfee.gif" width="40%"> <br>



## Citation 


```
@article{kwak2022injecting,
  author    = {Kwak, Jeong-gi and Li, Yuanming and Yoon, Dongsik and Kim, Donghyeon and Han, David and Ko, Hanseok},
  title     = {Injecting 3D Perception of Controllable NeRF-GAN into StyleGAN for Editable Portrait Image Synthesis},
  journal   = {arXiv},
  year      = {2022},
}
```

## Acknowledgments

-  SURF-GAN is bulided upon the [pi-GAN](https://github.com/marcoamonteiro/pi-GAN) implementation and inspired by [EigenGAN](https://github.com/LynnHo/EigenGAN-Tensorflow).  
-  We used [pSp encoder](https://github.com/eladrich/pixel2style2pixel) ([e4e](https://github.com/omertov/encoder4editing) also works) and [StyleGAN2-pytorch](https://github.com/rosinality/stylegan2-pytorch) to build 3D-controllable StyleGAN.  


