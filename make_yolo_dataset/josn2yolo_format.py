"""
本脚本有两个功能：
1.将josn数据集标注信息(.xml)转为yolo标注格式(.txt)，并将图像文件复制到相应文件夹
2.根据json标签文件，生成对应names标签(my_data_label.names)

修改
1.路径文件，保证原数据路径文件和待保存数据文件路径一致
2.从json文件中读取数据时，不同的数据集要具体分析
"""
# VOC数据集的root
import json
import os
import shutil

import cv2
from lxml import etree
from tqdm import tqdm
# 原始数据集路径
dataset_root = "../datasets/cs_datasets"
json_path="../datasets/cs_datasets/josns"
images_path="../datasets/cs_datasets/images"

# datset 所对应的josn文件，保存着种类所代表是什么
class_json_path = "../datasets/cs_datasets/cs_classes.json"

# /********************************************************************/

# 检查文件夹/文件是否存在
assert os.path.exists(dataset_root), "原数据集 总路径不存在"
assert os.path.exists(json_path), "原数据集 json路径不存在"
assert os.path.exists(images_path), "原数据集图像路径不存在"
assert os.path.exists(class_json_path), "原数据集 类别所对应的标注文件不存在"




def copy_file(from_path: str, to_path_dir: str, to_path_filename: str):
    # from_path:
    # to_path_dir:
    # to_path_filename:
    # 将图片复制到指定位置

    # 获得 目的路径完整地址
    to_path = os.path.join(to_path_dir, to_path_filename).replace("\\","/")#linux只能用/
    # 检查form路径存在性
    assert os.path.exists(from_path), "from路径的文件不存在"
    # 如果目的历经文件夹不存在，则创建文件夹
    if not os.path.exists(to_path_dir):
        os.mkdir(to_path_dir)
    # 如果目的路径文件夹存在，若文件不存在则正常写入
    if not os.path.exists(to_path_dir):
        # 调用文件操作库shutil
        shutil.copyfile(from_path, to_path)


def translate_voc2yolo(data_file: list,  class_dict: dict):
    """
    将voc xml文件标注格式转换为yolo格式
    :param data_file: 用list保存，每个元素为一张图片名字(无后缀)
    :param save_root: yolo数据集要保存的未知
    :param class_dict: 数据集的类别字典，[1,20]
    :return:
    """
    # 创建labels文件，保存yolo格式数据
    label_dir_path=os.path.join(dataset_root, "labels")
    if not os.path.exists(label_dir_path):
        os.makedirs(label_dir_path)
    #  2.挨个读取json 文件
    # tqdm()
    # args：
    #       iterable 为迭代序列
    #       desc 信息展示的前缀
    for filename in tqdm(iterable=data_file, desc="正在将{}集合从Josn 解析到 yolo 格式"):
        # 1. 检查下json文件是否存在
        josn_file_path = os.path.join(json_path, filename + ".json").replace("\\","/")#linux只能用/
        assert os.path.exists(josn_file_path), "file:{} not exist...".format(josn_file_path)
        #    检查下img文件是否存在
        img_file_path = os.path.join(images_path, filename + ".png").replace("\\","/")#linux只能用/
        assert os.path.exists(img_file_path), "file:{} not exist...".format(josn_file_path)
        # 2. 读取json文件
        with open(josn_file_path, "r") as josn_file:
            # 读取注元素
            data = json.load(josn_file)["shapes"]
        # 不存在对象
        if len(data) == 0:
            # 如果xml文件中没有目标就直接忽略该样本
            print("Warning: 在 '{}' json文件中无对象.".format(josn_file_path))
            continue

        # 3.读取data下的size 和heigth和width返回图像的大小
        #   用于4中计算yolo格式
        img = cv2.imread(img_file_path)
        img_height = int(img.shape[0])
        img_width = int(img.shape[1])

        # 4.循环获取 josn中的物体 将其转化为 yolo格式
        # 打开要将yolo标注信息保存的文件
        # 1.每一个object 代表一个物体 ，提取其 bndbox 获得边框,获得其分类编号
        with open(os.path.join(label_dir_path, filename + ".txt").replace("\\","/"), "w") as f:#linux只能用/
            for index, obj in enumerate(data):
                # 获取每一个object的box信息
                xmin = float(obj["points"][0][0])
                ymin = float(obj["points"][0][1])
                xmax = float(obj["points"][1][0])
                ymax = float(obj["points"][1][1])
                # 获得类名
                class_name = obj["label"]


                # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                if xmax <= xmin or ymax <= ymin:
                    print("Warning: 在 '{}' , 存在box w/h <=0".format(josn_file_path))
                    print("xmin:{},ymin:{},xmax:{},ymax:{}".format(xmin,ymin,xmax,ymax))
                    print("class_name:{}".format(class_name))
                    continue

                # 将 box 格式转化为 yolo 格式 , 绝对坐标
                xcenter = (xmax + xmin) / 2
                ycenter = (ymax + ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                # yolo 保存相对坐标,保留6为小数
                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                # 读取 类名 ，将类名对应到id ，由于没有背景类，所以id 从0开始
                class_index = class_dict[class_name] - 1

                # 将信息保存到打开的文件当中,index是循环扫描object的序号

                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

def main():
    # 1.读取文件信息
    # json加载类别信息
    with open(class_json_path, "r") as class_json:
        class_dict = json.load(class_json)  # 读取txt所有行，每一行为list的一个元素
        print("将voc格式数据对应其类的index为：{}".format(class_dict))
    # 读取指定文件夹下的 所有image 文件名，通过image名生成json文件路径
    image_name_list=[ os.path.splitext(image_name)[0] for image_name in os.listdir(images_path)]
    translate_voc2yolo(data_file=image_name_list, class_dict=class_dict)



if __name__ == "__main__":
    main()
