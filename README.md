# Single View Stereo Matching

<img src="figs/result.png" width="1200" height="250" />

This repo includes the source code of the paper:
["Single View Stereo Matching"](https://arxiv.org/abs/1803.02612) (CVPR'18 Spotlight) by Yue Luo*, [Jimmy Ren](http://www.jimmyren.com/)*, Mude Lin, Jiahao Pang, Wenxiu Sun, Hongsheng Li, Liang Lin.

Contact: Yue Luo (lawy623@gmail.com)

### Prerequisites
The code is tested on 64 bit Linux (Ubuntu 14.04 LTS). You should also install Matlab (We have tested on R2015a). We have tested our code on GTX TitanX with CUDA8.0+cuDNNv5. Please install all these prerequisites before running our code.
   
### Installation
1. Get the code. 
   ```Shell
   git clone https://github.com/lawy623/SVS.git
   cd SVS
   ```
2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.

   ```Shell
   cd caffe/
   # Modify Makefile.config according to your Caffe installation/. Remember to allow CUDA and CUDNN.
   make -j8
   make matcaffe
   ```
3. Prepare data. We write all data and labels into `.mat` files.

- Please go to directory `data/`, and run `get_data.sh` to download [Kitti Stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) and [Kitti Raw](http://www.cvlibs.net/datasets/kitti/raw_data.php) datasets.
- To create the `.mat` files, please go to directory `data`, and run the matlab scripts `prepareTrain.m` and `prepareTest.m` respectively. It will take some time to prepare data.
- If you only want to test our models, you can simply downloads the Eigen test file at \[[GoogleDrive](https://drive.google.com/file/d/1LLb0-SEkPzI1TDy5BsmzTBQu0ggkc9ZH/view?usp=sharing)|[BaiduPan](https://pan.baidu.com/s/1h3eXqFhmcVhlV4Jbs4BIYQ)\]. Put this test .mat file in `/data/testing/`.

### Training

#### View Synthesis Network
- As described in our paper, we develop our View Synthesis Network based on the [Deep3D](https://github.com/piiswrong/deep3d) method. Directly training the final model indicated in our paper using VGG16 initialization will easily sink into local optimum. We first keep the BatchNorm layers and train the model with VGG16 initialization. Go to `training/` to run `train_viewSyn.m`. You can also run the matlab scripts from terminal at directory `training/` by following commands. By default matlab is installed under `/usr/local/MATLAB/R2015a`. If the location of your matlab is not the same, please modify `train_ViewSyn.sh` if want to run the scripts from terminal. Download the VGG16 at \[[GoogleDrive](https://drive.google.com/file/d/1ZpZG9L3FwTjkXgy6sKkqADo-NJ2LHEDY/view?usp=sharing)|[BaiduPan](https://pan.baidu.com/s/1pC0-5MSoaFX6Arm4LSWV2w)\], and put it under `training/prototxt/viewSynthesis_BN/preModel/` before finetuning. We train such a BN model for roughly 30k iterations.

```Shell
   ## To run the training matlab scripts from terminal
   sh prototxt/viewSynthesis/train_ViewSyn.sh   #To trained the view synthesis network
```
-  We further remove the Batch Norm layers and obtain a better performance. Rename the trained BN model (in `training/prototxt/viewSynthesis_BN/caffemodel`) mentioned above as `viewSyn_BN.caffemodel`. Or you can directly download ours at \[[GoogleDrive](https://drive.google.com/)|[BaiduPan](https://pan.baidu.com/s/1uK5qWhAzglZS9Fs90IVShg)\] and place it in the correct place. Change `line 11` of `train_viewSyn.m` to be `‘model = param.model(2);’`， and run `train_viewSyn.m` again.

#### Stereo Matching Network
- We do not provide the training code for training this stereo matching network. We follow [CRL](https://github.com/Artifineuro/crl) and use their trained model. Relevant model settings can be found in `training/prototxt/stereo/`. 

#### Single View Stereo Matching - End-to-end finetune.
- To finetune our svs model, please first download the pretrain models for two sub-networks.
  Download View Synthesis Network at \[[GoogleDrive](https://drive.google.com)|[BaiduPan](https://pan.baidu.com/s/1R_xUE4zsyhW2QEyz-KynBg)\], and put it under `training/prototxt/viewSynthesis/caffemodel/`.
  Download Stereo Matching Network. You can download the model trained on FlyingThings synthetic dataset at \[[GoogleDrive](https://drive.google.com/file/d/1ItR4KFgKrYuf3elEweES6WPrSNjS3UaH/view?usp=sharing)|[BaiduPan](https://pan.baidu.com/s/1p2onPQSD6Rbm98Q-XQtzdg)\], and a model further finetuned on Kitti Stereo 2015 at \[[GoogleDrive](https://drive.google.com/file/d/1vo3Yw7e0QATinfmkOuvm7MY18eLy2vQk/view?usp=sharing)|[BaiduPan](https://pan.baidu.com/s/1crBgBXwz8YdxK8jju2S26A)\]. Put the downloaded models under `training/prototxt/stereo/caffemodel/`
- Go to `training/` to run `train_svs.m`. You can also run the matlab scripts from terminal at directory `training/` by following commands.  
```Shell
   ## To run the training matlab scripts from terminal
   sh prototxt/svs/train_svs.sh   #To trained the svs network
```
  
### Testing
- Downloads the Eigen test file at \[[GoogleDrive](https://drive.google.com/file/d/1LLb0-SEkPzI1TDy5BsmzTBQu0ggkc9ZH/view?usp=sharing)|[BaiduPan](https://pan.baidu.com/s/1h3eXqFhmcVhlV4Jbs4BIYQ)\]. Put this test .mat file in `/data/testing/`. Or you can follow the data preparation step mentioned above.
 Download svs model at \[[GoogleDrive](https://drive.google.com)|[BaiduPan](https://pan.baidu.com/s/1Er0I27-4tXqYCuesW2DwLw)\], and put it under `training/prototxt/svs/caffemodel/`.
- Go to directory `testing/`.
  Run `test_svs.m` to get the result before finetune. Please make sure to have downloaded the trained View Synthesis Network and Stereo Matching Network.
  Run `test_svs_end2end.m` to get our state-of-the-art result on monocular depth estimation.
- If you want to get some visible results, change `line 4` of `test_svs.m` or `test_svs_end2end.m` to be `‘visual = 1;’`.


### Results
- Some of our qualitative results are shown here.

<img src="figs/1.jpg" width="1000" height="1000" />

<img src="figs/2.jpg" width="1000" height="1000" />

<img src="figs/3.jpg" width="1000" height="1000" />

### Citation
Please cite our paper if you find it useful for your work:
```
@article{Luo2018SVS,
    title={Single View Stereo Matching},
    author={Yue Luo, Jimmy Ren, Mude Lin, Jiahao Pang, Wenxiu Sun, Hongsheng Li, Liang Lin},
    journal={arXiv preprint arXiv: 1803.02612},
    year={2018},
}


