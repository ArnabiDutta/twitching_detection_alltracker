# AllTracker: Efficient Dense Point Tracking at High Resolution

**[[Paper](https://arxiv.org/abs/2506.07310)] [[Project Page](https://alltracker.github.io/)]**

## Env setup

Install miniconda:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init
```

Set up a fresh conda environment for AllTracker:

```
conda create -n alltracker python=3.12.8
conda activate alltracker
pip install -r requirements.txt
```


## Running the demo

Download the model:

```
download_reference_model.sh
tar -xvf alltracker_reference.tar.gz
```

Download the sample video:
```
cd demo_video
sh download_video.sh
cd ..
```

Run the demo
```
python demo.py
```


## Training code

(Working on this...)

Download Kubric from [here].

This is just a torch export of the official `kubric-public/tfds/movi_f/512x512` data.

With Kubric, you can skip the other datasets and start training Stage 1.

Download the rest of the point tracking datasets from [here](https://huggingface.co/aharley/alltracker/tree/main).

There you will find 24-frame datasets, `ce24*.tar.gz`, and 64-frame datasets, `ce64*.tar.gz`. Some of the datasets are large, and they are split into parts, so you need to create the full files by concatenating. For example:
```
cat ce24_flt_aa ce24_flt_ab ce24_flt_ac ce24_flt_ad ce24_flt_ae > ce24_flt.tar.gz
```

Download the optical flow datasets from the official websites: [FlyingChairs, FlyingThings3D, Monkaa, Driving](https://lmb.informatik.uni-freiburg.de/resources/datasets) [AutoFlow](https://autoflow-google.github.io/), [SPRING](https://spring-benchmark.org/), [VIPER](https://playing-for-benchmarks.org/download/), [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/), [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow), [TARTANAIR](https://theairlab.org/tartanair-dataset/). 


# Stage 1

Stage 1 is to train the model for 200k steps on Kubric. 

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python train_stage1.py  --mixed_precision --lr 5e-4 --max_steps=200000 --data_dir /data --exp "stage1abc" 
```

This should produce a tensorboard log in `./logs_train/`, and checkpoints in `./checkponts/`, in folder names similar to "64Ai4i3_5e-4m_stage1abc_1318". (The 4-digit string at the end is a timecode indicating when the run began, to help make the filepaths unique.)


# Stage 2

Stage 2 is to train the model for 400k steps on a mix of point tracking datasets and optical flow datasets. This stage initializes from the output of Stage 1.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python train_stage2.py  --mixed_precision --init_dir='64Ai4i3_5e-4m_stage1abc_1318' --lr=1e-5 --max_steps=400000 --exp='stage2abc'

```



## Citation

If you use this code for your research, please cite:

```
Adam W. Harley, Yang You, Xinglong Sun, Yang Zheng, Nikhil Raghuraman, Yunqi Gu, Sheldon Liang, Wen-Hsuan Chu, Achal Dave, Pavel Tokmakov, Suya You, Rares Ambrus, Katerina Fragkiadaki, Leonidas J. Guibas. AllTracker: Efficient Dense Point Tracking at High Resolution. arXiv 2025.
```

Bibtex:
```
@inproceedings{harley2025alltracker,
author    = {Adam W. Harley and Yang You and Xinglong Sun and Yang Zheng and Nikhil Raghuraman and Yunqi Gu and Sheldon Liang and Wen-Hsuan Chu and Achal Dave and Pavel Tokmakov and Suya You and Rares Ambrus and Katerina Fragkiadaki and Leonidas J. Guibas},
title     = {All{T}racker: {E}fficient Dense Point Tracking at High Resolution}
booktitle = {arXiv},
year      = {2025}
}
```
