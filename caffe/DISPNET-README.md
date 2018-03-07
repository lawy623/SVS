Caffe with Disp- and FlowNet
============================

This the relase of:
 - the CVPR 2016 versions of DispNet and DispNetCorr1D and
 - the ICCV 2015 versions of FlowNet and FlowNetC.

It comes as a fork of the current caffe master branch and with trained networks,
as well as examples to use or train them.


Compiling
=========

To get started with DispNet, first compile caffe, by configuring a

    "Makefile.config" (example given in Makefile.config.example)

then make with 

    $ make -j 5 all tools


Running 
=======

(this assumes you compiled the code sucessfully) 

E.g. go to this folder:

    cd ./dispnet-release/models/DispNet/

To try out DispNet on sample image pairs, run

    ./demo.py imgL_list.txt imgR_list.txt 

Similar demos can be found for other variants and FlowNets. 


Training
========

(this assumes you compiled the code sucessfully) 

First you need to download and prepare the training data. For that go to the data folder: 

    cd data 

Then run: 
    ./download.sh 
    ./make-lmdbs.sh 

(this will take some time and disk space) 

To train a network: e.g. go to this folder:
 
    cd ./dispnet-release/models/DispNet/

Then just run: 
    ./train.py 

NOTE: The trainig results here may be not 100% identical to those in the papers, since there we have used different loss schedules and sometimes initialized with previous experiments.   


License and Citation
====================

Please cite this paper in your publications if you use DispNet for your research:

    @inproceedings{MIFDB16,
      author       = "N. Mayer and E. Ilg and P. H{\"a}usser and P. Fischer and D. Cremers and A. Dosovitskiy and T. Brox",
      title        = "A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      month        = "June",
      year         = "2016",
      url          = "http://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16"
    }

Please cite this paper in your publications if you use FlowNet for your research:

    @InProceedings{DFIB15,
      author       = "A. Dosovitskiy and P. Fischer and E. Ilg and P. H{\"a}usser and C. Haz\ırba\ş and V. Golkov and P. v.d. Smagt and D. Cremers and T. Brox",
      title        = "FlowNet: Learning Optical Flow with Convolutional Networks",
      booktitle    = "IEEE International Conference on Computer Vision (ICCV)",
      month        = "Dec",
      year         = "2015",
      url          = "http://lmb.informatik.uni-freiburg.de//Publications/2015/DFIB15"
    }
    
