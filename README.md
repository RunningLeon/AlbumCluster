# Album Clustering

## Brief Introduction

This repositiory contains a tool to cluster albums by face.
The main procedures are:

- Input an ablum, which has photos of mutiple people.
- Detect all face bounding boxes and five facial landmarks using [RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace).
- Align all cropped face images using five facial landmarks.
- Extract all 512-D face features using [insightface](https://github.com/deepinsight/insightface).
- Cluster all face features with [Chinese Whispers](https://en.wikipedia.org/wiki/Chinese_Whispers_(clustering_method)) and output a clustered album.

## How to use this tool?

### Clone Repository

```bash
git clone git@github.com:RunningLeon/AlbumCluster.git
```

### Create Enviroment

Use conda to manage python environment.

```bash
conda create -n env python==3.8.0
source activate env
pip install -r requirements.txt
```

**mxnet**: Support 1.2~1.6
First confirm the cuda version on your system, than use `pip` to install.
Example for cuda==10.2

```bash
pip install mxnet-cu102==1.4
```

**Compile rcnn**:

```bash
cd AlbumCluster/app/retinaface
make
```

### Run

```bash
python run.py --input path_to_album --output path_to_result
```

After a few minutes, results would be saved to `path_to_result`.

## FAQs

- None

## References

-
