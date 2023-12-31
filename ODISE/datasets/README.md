# Prepare Datasets for ODISE

Dataset preparation for ODISE follows [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md) and [Mask2Former](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md). 

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

ODISE has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable `DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```bash
$DETECTRON2_DATASETS/
  ade/
  coco/
  VOCdevkit/
  pascal_ctx_d2/
  pascal_voc_d2/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` under the ODISE project directory.

The [model zoo](../README.md#model-zoo)
contains configs and models that use these builtin datasets.


## Expected dataset structure for [COCO](https://cocodataset.org/#download):

```bash
coco/
  annotations/
    instances_{train,val}2017.json
    panoptic_{train,val}2017.json
    captions_{train,val}2017.json
    # below are prepare_coco_caption.py
    panoptic_caption_{train,val}2017.json  
  {train,val}2017/
  panoptic_{train,val}2017/ 
  # below are generated by prepare_coco_semantic_annos_from_panoptic_annos.py
  panoptic_semseg_{train,val}2017/  # 
```

Download the dataset from http://cocodataset.org/#download:
```bash
cd $DETECTRON2_DATASETS
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d coco/
wget http://images.cocodataset.org/zips/val2017.zip 
unzip val2017.zip -d coco/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d coco/
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip -d coco/
unzip coco/annotations/panoptic_train2017.zip -d coco/
unzip coco/annotations/panoptic_val2017.zip -d coco/
```

Install the panopticapi (also automatically done by installing ODISE) by:
```bash
pip install git+https://github.com/cocodataset/panopticapi.git
```

Generate the semantic segmentation annotations `coco/panoptic_semseg_{train,val}2017/` by running:
```bash
python datasets/prepare_coco_semantic_annos_from_panoptic_annos.py
``` 
to extract the semantic annotations from the panoptic ones (only used for evaluation).

Generate the panoptic annotations with COCO captions `panoptic_caption_{train,val}2017.json` by running:
```bash
python datasets/prepare_coco_caption.py
```


## Expected dataset structure for [ADE20k (A-150)](http://sceneparsing.csail.mit.edu/) and [ADE20k-Full (A-847)](https://groups.csail.mit.edu/vision/datasets/ADE20K/):
```bash
ade/
  ADEChallengeData2016/
    images/
    annotations/
    objectInfo150.txt
    # downloaded instance annotation
    annotations_instance/
    # generated by prepare_ade20k_sem_seg.py
    annotations_detectron2/
    # generated by prepare_ade20k_ins_seg.py
    ade20k_instance_{train,val}.json
    # generated by prepare_ade20k_pan_seg.py
    ade20k_panoptic_{train,val}.json
    ade20k_panoptic_{train,val}/
    
  ADE20K_2021_17_01/
    images/
    index_ade20k.pkl
    objects.txt
    # generated by prepare_ade20k_full_sem_seg.py
    images_detectron2/
    annotations_detectron2/
    
```

### ADE20k(A-150)

Download the dataset from http://sceneparsing.csail.mit.edu/:
```bash
cd $DETECTRON2_DATASETS
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
# generate folder ade/ADEChallengeData2016/
unzip ADEChallengeData2016.zip -d ade/
```

Download the instance annotations from http://sceneparsing.csail.mit.edu/:
```bash
cd $DETECTRON2_DATASETS
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
# generate folder ade/ADEChallengeData2016/annotations_instance/
tar -xvf annotations_instance.tar -C ade/ADEChallengeData2016/
```

Generate the directory `ade/ADEChallengeData2016/annotations_detectron2` by running: 
```bash
python datasets/prepare_ade20k_sem_seg.py
```

Generate instance annotations `ade/ADEChallengeData2016/ade20k_instance_{train,val}.json` by running: 
```bash
python datasets/prepare_ade20k_ins_seg.py
```

Generate panoptic annotations `ade/ADEChallengeData2016/ade20k_panoptic_{train,val}.json` and `ade/ADEChallengeData2016/ade20k_panoptic_{train,val}` by running: 
```bash
python datasets/prepare_ade20k_pan_seg.py
```

### ADE20k-Full(A-847)

Register and download the dataset from https://groups.csail.mit.edu/vision/datasets/ADE20K/:
```bash
cd $DETECTRON2_DATASETS
wget your/personal/download/link/{username}_{hash}.zip
unzip {username}_{hash}.zip -d ade/
```

Generate the directories `ade/ADE20K_2021_17_01/images_detectron2` and `ade/ADE20K_2021_17_01/annotations_detectron2` by running: 
```bash
python datasets/prepare_ade20k_full_sem_seg.py
```

## Expected dataset structure for [PASCAL Context (PC-59)](https://www.cs.stanford.edu/~roozbeh/pascal-context/), [PASCAL Context Full (PC-459)](https://www.cs.stanford.edu/~roozbeh/pascal-context/) and [PASCAL VOC (PAS-21)](http://host.robots.ox.ac.uk/pascal/VOC/):

```bash
VOCdevkit/
  VOC2012/
    Annotations/
    JPEGImages/
    ImageSets/
      Segmentation/
  VOC2010/
    JPEGImages/
    trainval/
    trainval_merged.json
# generated by prepare_pascal_voc_sem_seg.py
pascal_voc_d2/
  images/
  annotations_pascal21/
# generated by prepare_pascal_ctx_sem_seg.py
pascal_ctx_d2/
  images/
  annotations_ctx59/
  # generated by prepare_pascal_ctx_full_sem_seg.py
  annotations_ctx459/

```
### PASCAL VOC (PAS-21)

Download the dataset from http://host.robots.ox.ac.uk/pascal/VOC/:
```bash
cd $DETECTRON2_DATASETS
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# generate folder VOCdevkit/VOC2012
tar -xvf VOCtrainval_11-May-2012.tar
```

Generate directory `pascal_voc_d2` running: 
```bash
python datasets/prepare_pascal_voc_sem_seg.py
```

### PASCAL Context (PC-59)

Download the dataset from http://host.robots.ox.ac.uk/pascal/VOC/ and annotation from https://www.cs.stanford.edu/~roozbeh/pascal-context/:
```bash
cd $DETECTRON2_DATASETS
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
# generate folder VOCdevkit/VOC2010
tar -xvf VOCtrainval_03-May-2010.tar 
wget https://www.cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz
# generate folder VOCdevkit/VOC2010/trainval
tar -xvzf trainval.tar.gz -C VOCdevkit/VOC2010 
wget https://codalabuser.blob.core.windows.net/public/trainval_merged.json -P VOCdevkit/VOC2010/
```

Install [Detail API](https://github.com/zhanghang1989/detail-api) by:
```bash
git clone https://github.com/zhanghang1989/detail-api.git
rm detail-api/PythonAPI/detail/_mask.c
pip install -e detail-api/PythonAPI/
```

Generate directory `pascal_ctx_d2/images` and `pascal_ctx_d2/annotations_ctx59` running:
```bash
python datasets/prepare_pascal_ctx_sem_seg.py
```

### PASCAL Context Full (PC-459)

Generate directory `pascal_ctx_d2/annotations_ctx459` running:
```bash
python datasets/prepare_pascal_ctx_full_sem_seg.py
```
