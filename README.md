# Injecting 3D Perception of Controllable NeRF-GAN into StyleGAN for Editable Portrait Image Synthesis

<img src="https://user-images.githubusercontent.com/67986601/191766555-806a304e-25a6-4d7d-9d48-ce623febe85f.png" width="90%">

## [Project page](https://jgkwak95.github.io/surfgan/) | [Paper](http://arxiv.org/abs/2207.10257)  
<br>
  
>**"Injecting 3D Perception of Controllable NeRF-GAN into StyleGAN for Editable Portrait Image Synthesis"** <br>
>[Jeong-gi Kwak](https://jgkwak95.github.io/), Yuanming Li, Dongsik Yoon, Donghyeon Kim, David Han, Hanseok Ko<br>
>**ECCV 2022** <br>

<br>

This repository includes the official Pytorch implementation of SURF-GAN. <br>


# SURF-GAN 

SURF-GAN, which is a NeRF-based 3D-aware GAN, can discover disentangled semantic attributes in an unsupervised manner.  <br><br>
<img src="https://user-images.githubusercontent.com/67986601/180636584-6ac7f46f-ff8f-4da1-a4b9-69e31e769909.gif" width=150> <br>
(Tranined on 64x64 CelebA and rendered with 256x256) <br>

## Get started

- #### Clone the repo.
```
git clone https://github.com/jgkwak95/SURF-GAN.git
cd SURF-GAN
```
- #### Create virtual environment
```
conda create -n surfgan python=3.7.1
conda activate surfgan
conda install -c pytorch-lts pytorch torchvision 
pip install --no-cache-dir -r requirements.txt
```

## Train SURF-GAN
At first, look curriculum.py and specify dataset and training options.
```
# CelebA
python train_surf.py --output_dir your-exp-name \
--curriculum CelebA_single
```
### Pretrained model
Or, you can use the [pretrained model](https://drive.google.com/file/d/1twtS5-9CVzSEiLleY7qvJYd-R6G8f_Cg/view?usp=sharing).
## Semantic attribute discovery
Let's traverse each dimension with discovered semantics:
```
python discover_semantics.py  --experiment your-exp-name \
--image_size 256 \
--ray_step_multiplier 2 \
--num_id 9 \          
--traverse_range 3.0 \    
 --intermediate_points 9 \
--curriculum CelebA_single     
```
The default ckpt file to traverse is the latest file (generator.pth).
If you want to check specific cpkt, add this in your command line, for example,
```
--specific_ckpt 140000_64_generator.pth
```
## Control pose
In addition, you can control only camera paramters:
```
python control_pose.py --experiment your-exp-name \
--image_size 128 \
--ray_step_multiplier 2 \
--num_id 9 \
--intermediate_points 9 \
--mode yaw \
--curriculum CelebA_single \
```
## Render video
- #### Moving camera
Set the mode: yaw, pitch, fov, etc.
You can also make your trajectory. 
```
python render_video.py  --experiment your-exp-name \
--image_size 128 \
--ray_step_multiplier 2 \
--num_frames 100 \
--curriculum CelebA_single \
--mode yaw
```

- #### Moving camera with a specific semantic
Choose an attribute that you want to control LiDj.
```
python render_video_semantic.py  --experiment your-exp-name \
--image_size 128 \
--ray_step_multiplier 2 \
--num_frames 100 \ 
--traverse_range 3.0 \
--intermediate_points \
--curriculum CelebA_single \
--mode circle \
--L 2 \
--D 4
```
<br>
<br>

# 3D-Controllable StyleGAN

Injecting the prior of SURF-GAN into StyleGAN for controllable generation. <br>
Also, it is compatible with many StyleGAN-based methods.

<img src="https://jgkwak95.github.io/surfgan/assets/3d_stylegan.png" width="40%"> <br>

### Video

| Pose control  | + Style ([Toonify](https://github.com/justinpinkney/toonify)) | 
| ------ | ------| 
| <img src="https://user-images.githubusercontent.com/67986601/180636660-189a6479-084a-460f-ac69-a6491987d9b8.gif" width="90%"> | <img src="https://user-images.githubusercontent.com/67986601/180636724-0963eb04-5ec1-441c-8305-464c4796ef02.gif" width="100%">   |
| <img src="https://user-images.githubusercontent.com/67986601/180636672-4b22539a-73ee-470c-b5f8-795df3a27a5c.gif" width="90%">| <img src="https://user-images.githubusercontent.com/67986601/180636715-39a4438e-9176-4c3e-b8bd-b417c5b6d4c5.gif" width="100%"> |

<br>

It is capable of editing real images directly. (with [HyperStyle](https://github.com/yuval-alaluf/hyperstyle))


| Pose   | +Illumination (using SURF-GAN samples) | 
| ------ | ------| 
| <img src="https://user-images.githubusercontent.com/67986601/182040684-571dbfb7-833d-4729-84c1-e4c0544fffee.gif" width="100%"> | <img src="https://user-images.githubusercontent.com/67986601/182040360-0b295928-0e83-4b66-b284-05fb9711425c.gif" width="100%">   |

| +Hair color (using SURF-GAN samples)   | +Smile(using [InterFaceGAN](https://github.com/genforce/interfacegan)) | 
| ------ | ------| 
| <img src="https://user-images.githubusercontent.com/67986601/182040807-97c306a6-6a6a-4fa3-8dbc-12320f775283.gif" width="100%"> | <img src="https://user-images.githubusercontent.com/67986601/182040808-00d04350-dc7c-4499-aa90-fa35869ff642.gif" width="100%">   | <br>   
   
    
     
<br><br>
## Citation 


```
@article{kwak2022injecting,
  title={Injecting 3D Perception of Controllable NeRF-GAN into StyleGAN for Editable Portrait Image Synthesis},
  author={Kwak, Jeong-gi and Li, Yuanming and Yoon, Dongsik and Kim, Donghyeon and Han, David and Ko, Hanseok},
  journal={arXiv preprint arXiv:2207.10257},
  year={2022}
}
```

## Acknowledgments

-  SURF-GAN is bulided upon the [pi-GAN](https://github.com/marcoamonteiro/pi-GAN) implementation and inspired by [EigenGAN](https://github.com/LynnHo/EigenGAN-Tensorflow) ([EigenGAN-pytorch](https://github.com/bryandlee/eigengan-pytorch)). Thanks to the authors for their excellent work!
-  We used [pSp encoder](https://github.com/eladrich/pixel2style2pixel) and [StyleGAN2-pytorch](https://github.com/rosinality/stylegan2-pytorch) to build 3D-controllable StyleGAN. For editing  in-the-wild real images, we exploited [e4e](https://github.com/omertov/encoder4editing) and [HyperStyle](https://github.com/yuval-alaluf/hyperstyle) with our 3D-controllable StyleGAN. 

