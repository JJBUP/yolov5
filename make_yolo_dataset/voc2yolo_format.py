"""
本脚本有两个功能：
1.将voc数据集标注信息(.xml)转为yolo标注格式(.txt)，并将图像文件复制到相应文件夹
2.根据voc标签文件，生成对应names标签(my_data_label.names)

修改
1.路径文件，保证原数据路径文件和待保存数据文件路径一致
2.从xml文件中读取数据时，不同的数据集要具体分析
"""
# VOC数据集的root
import json
import os
import shutil

from lxml import etree
from tqdm import tqdm
# voc格式原数据文件
voc_root = "../datasets/none_yolo_dataset"

# Yolo格式文件的root
yoloDataset_root = "../datasets/yolo_dataset"

# 转换后的训练集以及验证集对应的txt文件
train_txt = "train.txt"
val_txt = "val.txt"

# label 所对应的josn文件，保存着1-20个种类所代表是什么
class_json_path = "../datasets/csgo/csgo_class.json"

#保存的label的names的文件
label_names_path = "../datasets/yolo_dataset/my_data_label.names"

# 拼接voc的目录
# 1.voc数据集 图像 目录
voc_image_path = os.path.join(voc_root, "JPEGImages")
# 2.标注文件
voc_xml_path = os.path.join(voc_root, "Annotations")
# 3.voc的 train val的分类目录
#    其中分为两类：
#       一类使按种类分类1/0/-1，易/难/无
#       另一类使按训练集和验证集合分类，代表训练和验证有哪些图片
voc_val_path = os.path.join(voc_root, "ImageSets", "Main", val_txt)
voc_train_path = os.path.join(voc_root, "ImageSets", "Main", train_txt)

# /********************************************************************/

# 检查文件夹/文件是否存在
assert os.path.exists(voc_image_path), "voc 图像路径不存在"
assert os.path.exists(voc_xml_path), "voc 标注路径不存在"
assert os.path.exists(voc_val_path), "voc test文件不存在"
assert os.path.exists(voc_train_path), "voc val文件不存在"
assert os.path.exists(class_json_path), "voc 类别所对应的标注文件不存在"
# 创建要保存的文件夹路径
if not os.path.exists(yoloDataset_root):
    os.makedirs(yoloDataset_root)


# 解析xml代码，将其保存为字典

def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    # 定义一个字典
    result = {}
    # 检查当前文件中是否有孩子节点
    for child in xml:
        # 执行树的后序遍历
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        # 1.如果是非object节点，即将object 中的所有child节点作为 保存为字典{key:value,key:value,key:value}
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        # 2.如果是object节点，将所有的object 保存为字典{key，list}
        else:
            # 处理object节点，保存为object的值为列表
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里（判断object是否在节点中（第一次），将其转化为列表形式）
                result[child.tag] = []
            result[child.tag].append(child_result[
                                         child.tag])  # 将annotation节点 所遍历的当前object的子节点（对象）加入object列表（返回的为字典，用child_result[“object”]取object的值
    return {xml.tag: result}


def copy_file(from_path: str, to_path_dir: str, to_path_filename: str):
    # from_path:
    # to_path_dir:
    # to_path_filename:
    # 将图片复制到指定位置

    # 获得 目的路径完整地址
    to_path = os.path.join(to_path_dir, to_path_filename)
    # 检查form路径存在性
    assert os.path.exists(from_path), "from路径的文件不存在"
    # 如果目的历经文件夹不存在，则创建文件夹
    if not os.path.exists(to_path_dir):
        os.mkdir(to_path_dir)
    # 如果目的路径文件夹存在，若文件不存在则正常写入
    if not os.path.exists(to_path_dir):
        # 调用文件操作库shutil
        shutil.copyfile(from_path, to_path)


def translate_voc2yolo(data_file: list, save_root: str, class_dict: dict, isTrain=True):
    """
    将voc xml文件标注格式转换为yolo格式
    :param data_file: voc train 或 val ，用list保存，每个元素为一张图片名字
    :param save_root: yolo数据集要保存的未知
    :param class_dict: voc 数据集的类别字典，[1,20]
    :param isTrain: 是否是训练集
    :return:
    """
    # 1.创建yolo格式的labels 和 JPEGImages 文件夹

    label_dir_path = os.path.join(save_root, "train" if isTrain else "val", "label")
    image_dir_path = os.path.join(save_root, "train" if isTrain else "val", "image")
    if not os.path.exists(label_dir_path):
        os.makedirs(label_dir_path)
    if not os.path.exists(image_dir_path):
        os.makedirs(image_dir_path)

    #  2.挨个读取xml 文件
    # tqdm()
    # args：
    #       iterable 为迭代序列
    #       desc 信息展示的前缀
    for filename in tqdm(iterable=data_file,
                         desc="正在将{}集合从voc 解析到 yolo 格式".format(" \"训练集\" " if isTrain else " \"验证集\" ")):
        # 1. 检查下xml文件是否存在
        xml_file_path = os.path.join(voc_xml_path, filename + ".xml")
        assert os.path.exists(xml_file_path), "file:{} not exist...".format(xml_file_path)
        # 检查 当前读取的对应的 img文件是否存在
        img_file_path = os.path.join(voc_image_path, filename + ".jpg")
        assert os.path.exists(img_file_path), "file:{} not exist...".format(img_file_path)

        # 2. 读取xml文件
        with open(xml_file_path, "r") as xml_file:
            xml_str = xml_file.read()
        # 使用lxml包下的etree来读取xml文件，返回xml的根节点
        xml = etree.fromstring(xml_str)
        # 将xml解压为字典形式，并读取 annotation 根节点字典对象
        data = parse_xml_to_dict(xml)["annotation"]

        # 不存在object key的情况
        assert "object" in data.keys(), "读取的xml文件: '{}' 缺少object 标签.".format(xml_file_path)
        # 存在object key 但是长度为0
        if len(data["object"]) == 0:
            # 如果xml文件中没有目标就直接忽略该样本
            print("Warning: 在 '{}' xml文件中没有object标签.".format(xml_file_path))
            continue

        # 3.读取data下的size 和heigth和width返回图像的大小
        #   用于4中计算yolo格式
        img_height = int(data["size"]["height"])
        img_width = int(data["size"]["width"])

        # 4.循环获取 xml 的object 将其转化为 yolo格式
        # 打开要将yolo标注信息保存的文件
        # 1.每一个object 代表一个物体 ，提取其 bndbox 获得边框,获得其分类编号
        with open(os.path.join(label_dir_path, filename + ".txt"), "w") as f:
            for index, obj in enumerate(data["object"]):
                # 获取每一个object的box信息
                xmin = float(obj["bndbox"]["xmin"])
                ymin = float(obj["bndbox"]["ymin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymax = float(obj["bndbox"]["ymax"])
                # 获得类名
                class_name = obj["name"]
                # 找到类名对应的编号
                class_index = class_dict[class_name]

                # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                if xmax <= xmin or ymax <= ymin:
                    print("Warning: 在 '{}' xml, 存在box w/h <=0".format(xml_file_path))
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

            # 5.将image文件一起复制过去
            # 拼接新路径的文件地址os.sep为“/”
            path_copy_to = os.path.join(image_dir_path, img_file_path.split(os.sep)[-1])
            if not os.path.exists(path_copy_to):
                shutil.copyfile(img_file_path, path_copy_to)


def create_class_names(class_dict: dict):
    keys = class_dict.keys()
    with open(label_names_path, "w") as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + "\n")


def main():
    # 1.读取文件信息
    # json加载类别信息
    with open(class_json_path, "r") as class_json:
        class_dict = json.load(class_json)  # 读取txt所有行，每一行为list的一个元素
        print("将voc格式数据对应其类的index为：{}".format(class_dict))
    # 2. 读取train.txt 和 val.txt 文件
    # 验证集转换
    with open(voc_val_path, "r") as voc_val_txt:
        # 读取整个文件夹，去除回车符，去掉空格
        voc_val = list(item.strip() for item in voc_val_txt.readlines() if len(item.strip()) > 0)
        # print(voc_val)
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_voc2yolo(data_file=voc_val, save_root=yoloDataset_root, class_dict=class_dict, isTrain=False)

    # 训练集转换
    with open(voc_train_path, "r") as voc_train_txt:
        voc_train = list(item.strip() for item in voc_train_txt.readlines() if len(item.strip()) > 0)
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_voc2yolo(data_file=voc_train, save_root=yoloDataset_root, class_dict=class_dict, isTrain=True)
    #

    create_class_names(class_dict)


if __name__ == "__main__":
    main()
