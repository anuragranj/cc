# Competitive Collaboration
This is an official repository of
**Competitive Collaboration: Joint Unsupervised Learning of Depth, Camera Motion, Optical Flow and Motion Segmentation**. The project was formerly referred by **Adversarial Collaboration**. We recently ported the entire code to `pytorch-1.0`, so if you discover bugs, please file an issue.

[[Project Page]](http://research.nvidia.com/publication/2018-05_Adversarial-Collaboration-Joint)
[[Arxiv]](https://arxiv.org/abs/1805.09806)

Skip to:
- [Joint Unsupervised Learning of Depth, Camera Motion, Optical Flow and Motion Segmentation](#jointcc)
- [Mixed Domain Learning using MNIST+SVHN](#mnist)
- [Download Pretrained Models and Evaluation Data](#downloads)

### Prerequisites
Python3 and pytorch are required. Third party libraries can be installed (in a `python3 ` virtualenv) using:

```bash
pip3 install -r requirements.txt
```
<a name="jointcc"></a>
## Joint Unsupervised Learning of Depth, Camera Motion, Optical Flow and Motion Segmentation

### Preparing training data

#### KITTI
For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command.

```bash
python3 data/prepare_train_data.py /path/to/raw/kitti/dataset/ --dataset-format 'kitti' --dump-root /path/to/resulting/formatted/data/ --width 832 --height 256 --num-threads 1 --static-frames data/static_frames.txt --with-gt
```

For testing optical flow ground truths on KITTI, download [KITTI2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) dataset. You need to download 1) `stereo 2015/flow 2015/scene flow 2015` data set (2 GB), 2) `multi-view extension` (14 GB), and 3) `calibration files` (1 MB) . In addition, download semantic labels from [here](https://keeper.mpdl.mpg.de/f/239c2dda94e54c449401/?dl=1). You should have the following directory structure:
```
kitti2015
  | data_scene_flow  
  | data_scene_flow_calib
  | data_scene_flow_multiview  
  | semantic_labels
```

#### Cityscapes

For [Cityscapes](https://www.cityscapes-dataset.com/), download the following packages: 1) `leftImg8bit_sequence_trainvaltest.zip`, 2) `camera_trainvaltest.zip`. You will probably need to contact the administrators to be able to get it.

```bash
python3 data/prepare_train_data.py /path/to/cityscapes/dataset/ --dataset-format 'cityscapes' --dump-root /path/to/resulting/formatted/data/ --width 832 --height 342 --num-threads 1
```

Notice that for Cityscapes the `img_height` is set to 342 because we crop out the bottom part of the image that contains the car logo, and the resulting image will have height 256.

### Training an experiment

Once the data are formatted following the above instructions, you should be able to run a training experiment. Every experiment you run gets logged in `experiment_recorder.md`.

```bash
python3 train.py /path/to/formatted/data --dispnet DispResNet6 --posenet PoseNetB6 \
  --masknet MaskNet6 --flownet Back2Future --pretrained-disp /path/to/pretrained/dispnet \
  --pretrained-pose /path/to/pretrained/posenet --pretrained-flow /path/to/pretrained/flownet \
  --pretrained-mask /path/to/pretrained/masknet -b4 -m0.1 -pf 0.5 -pc 1.0 -s0.1 -c0.3 \
  --epoch-size 1000 --log-output -f 0 --nlevels 6 --lr 1e-4 -wssim 0.997 --with-flow-gt \
  --with-depth-gt --epochs 100 --smoothness-type edgeaware  --fix-masknet --fix-flownet \
  --log-terminal --name EXPERIMENT_NAME
```


You can then start a `tensorboard` session in this folder by
```bash
tensorboard --logdir=checkpoints/
```
and visualize the training progress by opening [https://localhost:6006](https://localhost:6006) on your browser.

### Evaluation

Disparity evaluation
```bash
python3 test_disp.py --dispnet DispResNet6 --pretrained-dispnet /path/to/dispnet --pretrained-posent /path/to/posenet --dataset-dir /path/to/KITTI_raw --dataset-list /path/to/test_files_list
```

Test file list is available in kitti eval folder. To get fair comparison with [Original paper evaluation code](https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py), don't specify a posenet. However, if you do,  it will be used to solve the scale factor ambiguity, the only ground truth used to get it will be vehicle speed which is far more acceptable for real conditions quality measurement, but you will obviously get worse results.

For pose evaluation, you need to download [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset.
```bash
python test_pose.py pretrained/pose_model_best.pth.tar --img-width 832 --img-height 256 --dataset-dir /path/to/kitti/odometry/ --sequences 09 --posenet PoseNetB6
```

Optical Flow evaluation
```bash
python test_flow.py --pretrained-disp /path/to/dispnet --pretrained-pose /path/to/posenet --pretrained-mask /path/to/masknet --pretrained-flow /path/to/flownet --kitti-dir /path/to/kitti2015/dataset
```

Mask evaluation
```bash
python test_mask.py --pretrained-disp /path/to/dispnet --pretrained-pose /path/to/posenet --pretrained-mask /path/to/masknet --pretrained-flow /path/to/flownet --kitti-dir /path/to/kitti2015/dataset
```

<a name="mnist"></a>
## Mixed Domain Learning using MNIST+SVHN

#### Training
For learning classification using Competitive Collaboration with two agents, Alice and Bob, run,
```bash
python3 mnist.py path/to/download/mnist/svhn/datasets/ --name EXP_NAME --log-output --log-terminal --epoch-size 1000 --epochs 400 --wr 1000
```

#### Evaluation
To evaluate the performance of Alice, Bob and Moderator trained using CC, run,
```bash
python3 mnist_eval.py path/to/mnist/svhn/datasets --pretrained-alice pretrained/mnist_svhn/alice.pth.tar --pretrained-bob pretrained/mnist_svhn/bob.pth.tar --pretrained-mod pretrained/mnist_svhn/mod.pth.tar
```

<a name="downloads"></a>
## Downloads
#### Pretrained Models
- [DispNet, PoseNet, MaskNet and FlowNet](https://keeper.mpdl.mpg.de/f/72e946daa4e0481fb735/?dl=1) in joint unsupervised learning of depth, camera motion, optical flow and motion segmentation.
- [Alice, Bob and Moderator](https://keeper.mpdl.mpg.de/f/d0c7d4ebd0d74b84bf10/?dl=1) in Mixed Domain Classification

#### Evaluation Data
- [Semantic Labels for KITTI](https://keeper.mpdl.mpg.de/f/239c2dda94e54c449401/?dl=1)

## Acknowlegements
We thank Frederik Kunstner for verifying the convergence proofs. We are grateful to Clement Pinard for his [github repository](https://github.com/ClementPinard/SfmLearner-Pytorch). We use it as our initial code base. We thank Georgios Pavlakos for helping us with several revisions of the paper. We thank Joel Janai for preparing optical flow visualizations, and Clement Gorard for his Make3d evaluation code.


## References
*Anurag Ranjan, Varun Jampani, Lukas Balles, Deqing Sun, Kihwan Kim, Jonas Wulff and Michael J. Black.*  **Competitive Collaboration: Joint unsupervised learning of depth, camera motion, optical flow and motion segmentation.** CVPR 2019.
