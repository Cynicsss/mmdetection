#!/usr/bin/env bash

python3 setup.py develop
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pycocotools
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple terminaltables
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple mmcv
#pip3 install mmcv
mkdir txtresults
mkdir data
cd data
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/datasets/2018origincoco.zip
unzip -q 2018origincoco.zip
mv ./coco/train2017 ./coco/traintotal
mv ./coco/train2019 ./coco/train2017
mv ./coco/annotations/instances_train2017.json ./coco/annotations/instances_traintotal.json
mv ./coco/annotations/instances_train2019.json ./coco/annotations/instances_train2017.json
cd ..
hdfs dfs -get $PAI_DEFAULT_FS_URI/data/models/zhangyu/resnext101_64x4d-ee2c6f71.pth
mv ./resnext101_64x4d-ee2c6f71.pth /root/.cache/torch/checkpoints/
bash tools/dist_train.sh configs/faster_rcnn_x101_64x4d_fpn_1x.py 2
