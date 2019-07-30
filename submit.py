from pypai import PAI

pai = PAI(username='zhangyu', passwd='sdojj8689075')

#zhangyu:1.3
pai.submit()

#test
# python3 tools/test .py configs/cascade_rcnn_x101_32x4d_fpn_1x.py work_dirs/cascade_rcnn_x101_32x 4d_fpn_1x/epoch_1.pth --out results.pkl --eval bbox
# mkdir txtresults