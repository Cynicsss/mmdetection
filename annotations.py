import cv2
import os
import matplotlib.cm
import torch

CLASS_NAMES = ("ign", "ped", "peo", "byc", "car", "van", "trk", "tcy", "atc", "bus", "mtr", "oth")


def visualize(img, annos, classnames=CLASS_NAMES, with_score=True, xywh=True):
    """
    Mark annotation bounding box on the image.
    :param img: cv2 image
    :param annos: array with annotations
    :param classnames: class name text
    :param with_score: If show the score of each bbox.
    :param xywh: if xywh
    :return: marked image
    """
    img = img.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    colors = load_colors()
    for anno in annos:
        if not isinstance(anno, torch.Tensor):
            anno = anno.strip().split(',')
        x, y, w, h, score, cls = \
            round(float(anno[0])), round(float(anno[1])), round(float(anno[2])), round(float(anno[3])), float(anno[4]), round(float(anno[5]))
        if xywh:
            cv2.rectangle(img, (x, y), (x + w, y + h), colors[cls], 1)
        else:
            cv2.rectangle(img, (x, y), (w, h), colors[cls], 1)

        if with_score:
            cv2.putText(img, "{:.2}".format(score), (x + 2, y + 8), font, 0.3, colors[cls], 1, False)

    for i in range(len(classnames)):
        cv2.rectangle(img, (i * 35, 0), ((i+1) * 35, 15), colors[i], thickness=-1)
        cv2.putText(img, classnames[i], (i * 35 + 5, 13), font, 0.4, (255, 255, 255), 1, False)

    return img


def load_colors(num=12):
    """
    Generate color maps.
    :param num: color numbers.
    :return: colors, RGB from 0 to 255
    """
    colors = matplotlib.cm.get_cmap('tab20').colors
    assert num < len(colors)
    colors = [(int(color[0]*255), int(color[1]*255), int(color[2]*255)) for color in colors]
    return colors[:num]


if __name__ == '__main__':
    i = 0
    '''
    for img_name in os.listdir('F:/dataset/UNDERALL/coco/val22017'):
        dev_img = cv2.imread('F:/dataset/UNDERALL/coco/val22017/'+img_name)
        with open('F:/dataset/UNDERALL/txtresults1/txtresults/'+str(i)+'.txt', 'r') as reader:
            dev_annos = reader.readlines()
        marked_img = visualize(dev_img, dev_annos)
        cv2.imwrite('F:/dataset/UNDERALL/resultimages/'+img_name, marked_img)
        i += 1
        # cv2.waitKey(0)
    '''
    dev_img = cv2.imread('F:/dataset/UNDERALL/mergeimage/21871.jpg')
    with open('F:/dataset/UNDERALL/mergechangetxt/21871.txt', 'r') as reader:
        dev_annos = reader.readlines()
    marked_img = visualize(dev_img, dev_annos)
    cv2.imwrite('F:/dataset/UNDERALL/a.jpg', marked_img)