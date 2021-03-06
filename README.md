# Album Clustering
![avatar](images/pipeline.png)
## Brief Introduction

This repository contains a tool to cluster albums by face.
The main procedures are:

- Input an album, which has photos of multiple people.
- Detect all face bounding boxes and five facial landmarks using [RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace).
- Align all cropped face images using five facial landmarks.
- Extract all 512-D face features using [insightface](https://github.com/deepinsight/insightface).
- Cluster all face features with [Chinese Whispers](https://en.wikipedia.org/wiki/Chinese_Whispers_(clustering_method)) and output a clustered album.

## How to use this tool?

### Clone Repository

```bash
git clone git@github.com:RunningLeon/AlbumCluster.git
export ALBUM_HOME=$(pwd)/AlbumCluster
```

### Create Environment

Strongly suggest use `conda` to manage Python environment.
If you do not have conda on your system, please goto download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html)

- create conda environment
Once `conda` is installed, please follow steps below to create the environment.

```bash
conda create -n env python==3.7.0
source activate env
pip install -r requirements.txt
```

- Install mxnet-gpu
By defaults, there is mxnet-cpu in [requirements.txt](./requirements.txt). If you have cuda installed on your system, you can install mxnet-gpu. Supported version is from 1.2 to 1.6.
First confirm the cuda version on your system, than use `pip` to install.
Example for cuda==10.2

```bash
pip install mxnet-cu102==1.6.0
```

- Compile rcnn

```bash
cd $ALBUM_HOME/AlbumCluster/app/retinaface
make
```

- Download models
Link: [Baidu YunPan](https://pan.baidu.com/s/1SQHu5fHB8SGB-7bJ0WmErQ)
Code: `fsfl`

Download `models.zip` and put it in `$AlbumCluster/models` directory.

```bash
cd $AlbumCluster/models
unzip models.zip
```

### Run

```bash
python run.py --input ./images --output ./output
```

After a few minutes, results would be saved to `output` directory.

## Example Results

![avatar](images/clusters.png)
## FAQs

- None

## References

- [RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace)
- [insightface](https://github.com/deepinsight/insightface)
- [Chinese Whispers](https://blog.csdn.net/u011808673/article/details/78644485/)