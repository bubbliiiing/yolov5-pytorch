import json
from xml.dom.minidom import *
import xml.etree.ElementTree as ET
import numpy as np

#-----------------------------------------------------------#
#
#             本文件用于将coco数据集中的json文件
#                转化为voc数据集用的xml标签
#-----------------------------------------------------------#
# 美化xml文件
def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作

# 写入xml文件
def write_xml(img_name, width, height, object_dicts, save_path, folder='XJTU_dataset'):
    '''
    object_dict = {'name': classes[int(object_category)],
                            'truncated': int(truncation),
                            'difficult': int(occlusion),
                            'xmin':int(bbox_left),
                            'ymin':int(bbox_top),
                            'xmax':int(bbox_left) + int(bbox_width),
                            'ymax':int(bbox_top) + int(bbox_height)
                            }
    '''
    doc = Document
    root = ET.Element('Annotation')
    ET.SubElement(root, 'folder').text = folder
    ET.SubElement(root, 'filename').text = img_name
    size_node = ET.SubElement(root, 'size')
    ET.SubElement(size_node, 'width').text = str(width)
    ET.SubElement(size_node, 'height').text = str(height)
    ET.SubElement(size_node, 'depth').text = '3'
    for object_dict in object_dicts:
        object_node = ET.SubElement(root, 'object')
        ET.SubElement(object_node, 'name').text = object_dict['name']
        ET.SubElement(object_node, 'pose').text = 'Unspecified'
        ET.SubElement(object_node, 'truncated').text = str(object_dict['truncated'])
        ET.SubElement(object_node, 'difficult').text = str(object_dict['difficult'])
        bndbox_node = ET.SubElement(object_node, 'bndbox')
        ET.SubElement(bndbox_node, 'xmin').text = str(object_dict['xmin'])
        ET.SubElement(bndbox_node, 'ymin').text = str(object_dict['ymin'])
        ET.SubElement(bndbox_node, 'xmax').text = str(object_dict['xmax'])
        ET.SubElement(bndbox_node, 'ymax').text = str(object_dict['ymax'])

    pretty_xml(root, '\t', '\n')
    tree = ET.ElementTree(root)
    tree.write(save_path, encoding='utf-8')

if __name__ == '__main__':
    # class_names = ['road', 'car', 'motorcycle', 'person']
    # f = open("../../../cow/train.json", encoding='utf-8')
    f = open("cowboyoutfits/train.json", encoding='utf-8')
    data = json.load(f)
    annotations = data['annotations']
    images = data['images']
    cate = {
        "87": "belt",
        "1034": "sunglasses",
        "131": "boot",
        "318": "cowboy_hat",
        "588": "jacket"
    }
    images_num = len(images)
    # print()
    bbox_infos = {}
    for image in images:
        print()
        bbox_infos[str(image["id"])] = []

    for ann in annotations:
        print(ann)
        image_id = ann['image_id']
        bbox = ann['bbox']
        bbox_x = bbox[0]
        bbox_y = bbox[1]
        bbox_w = bbox[2]
        bbox_h = bbox[3]
        class_id = ann['category_id']
        object_dict = {'name': cate[str(class_id)],
                       'truncated': 0,
                       'difficult': 0,
                       'xmin': int(bbox_x),
                       'ymin': int(bbox_y),
                       'xmax': int(bbox_x+bbox_w),
                       'ymax': int(bbox_y+bbox_h)
                       }
        print(object_dict)
        bbox_infos[str(image_id)].append(object_dict)


    txt_results = []
    for image in images:
        print(image)
        image_height = image['height']
        image_width = image['width']
        image_id = image['id']
        image_file_name = image['file_name']
        object_dicts = bbox_infos[str(image_id)]
        xml_file_name = image_file_name.strip('.jpg') + '.xml'
        txt_results.append(image_file_name.strip('.jpg'))
        # print(image_file_name, image_width, image_height, object_dicts, )
        write_xml(image_file_name, image_width, image_height, object_dicts, "D:/yolov5-pytorch/cowboy_voc_xml/"+xml_file_name)

    # np.savetxt()
    # np.savetxt('train.txt', txt_results, fmt="%s", delimiter="\n")
    # for image in images:
    #     print(image)