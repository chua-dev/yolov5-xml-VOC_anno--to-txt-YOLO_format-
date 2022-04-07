"""  take voc Dataset annotation information (.xml) To yolo annotation format (.txt), And copy the image file to the corresponding folder  """
import os
from tqdm import tqdm
from lxml import etree
import json
import shutil


# voc The root directory and version of the dataset 
voc_root = "../hardhat_data/data/HeadDataset/VOC2028"
voc_version = "VOC2028"

#  The transformed training set and verification set correspond to txt file 
train_txt = "train.txt"
val_txt = "val.txt"

#  Save the converted files in the directory 
save_file_root = "../hardhat_data/my_yolo_dataset"

# label The label corresponding json file 
#label_json_path = './data/Head_classes.json'

#  Put together voc Of images Catalog ,xml Catalog ,txt Catalog 
voc_images_path = os.path.join(voc_root, "JPEGImages")
voc_xml_path = os.path.join(voc_root, "Annotations")
train_txt_path = os.path.join(voc_root, "ImageSets", "Main", train_txt)
val_txt_path = os.path.join(voc_root, "ImageSets", "Main", val_txt)
print(voc_images_path)
#  Check the documents / Whether all folders exist 
assert os.path.exists(voc_images_path), "VOC images path not exist..."
assert os.path.exists(voc_xml_path), "VOC xml path not exist..."
assert os.path.exists(train_txt_path), "VOC train txt file not exist..."
assert os.path.exists(val_txt_path), "VOC val txt file not exist..."
#assert os.path.exists(label_json_path), "label_json_path does not exist..."
if os.path.exists(save_file_root) is False:
    os.makedirs(save_file_root)


def parse_xml_to_dict(xml):
    """  take xml The file is parsed into dictionary form , Reference resources tensorflow Of recursive_parse_xml_to_dict Args： xml: xml tree obtained by parsing XML file contents using lxml.etree Returns: Python dictionary holding XML contents. """

    if len(xml) == 0:  #  Traverse to the bottom , Go straight back to tag Corresponding information 
        return {
    xml.tag: xml.text}

    result = {
    }
    for child in xml:
        child_result = parse_xml_to_dict(child)  #  Recursive traversal of label information 
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  #  because object There may be more than one , So you need to put it on the list 
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {
    xml.tag: result}


def translate_info(file_names: list, save_root: str, class_dict: dict, train_val='train'):
    """  To the corresponding xml File information is converted to yolo Used in txt file information  :param file_names: :param save_root: :param class_dict: :param train_val: :return: """
    save_txt_path = os.path.join(save_root, train_val, "labels")
    if os.path.exists(save_txt_path) is False:
        os.makedirs(save_txt_path)
    save_images_path = os.path.join(save_root, train_val, "images")
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path)

    for file in tqdm(file_names, desc="translate {} file...".format(train_val)):
        #  Check whether the image file exists 
        img_path = os.path.join(voc_images_path, file + ".jpg")
        assert os.path.exists(img_path), "file:{} not exist...".format(img_path)

        #  Check xml Does the file exist 
        xml_path = os.path.join(voc_xml_path, file + ".xml")
        assert os.path.exists(xml_path), "file:{} not exist...".format(xml_path)

        # read xml
        with open(xml_path, encoding='UTF-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data["size"]["height"])
        img_width = int(data["size"]["width"])
        # write object info into txt
        assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
        if len(data["object"]) == 0:
            #  If xml If there is no target in the file, ignore the sample directly 
            print("Warning: in '{}' xml, there are no objects.".format(xml_path))
            continue

        with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
            for index, obj in enumerate(data["object"]):
                #  For each object Of box Information 
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"]
                class_index = class_dict[class_name] - 1  #  The goal is id from 0 Start 

                #  Further check the data , Some annotation information may contain w or h by 0 The situation of , Such data will lead to the calculation of regression loss by nan
                if xmax <= xmin or ymax <= ymin:
                    print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                    continue

                #  take box Information conversion to yolo Format 
                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                #  Absolute coordinate to relative coordinate , preservation 6 Decimal place 
                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

        # copy image into save_images_path
        path_copy_to = os.path.join(save_images_path, img_path.split(os.sep)[-1])
        if os.path.exists(path_copy_to) is False:
            shutil.copyfile(img_path, path_copy_to)


def main():
    # read class_indict
    #json_file = open(label_json_path, 'r')
    #class_dict = json.load(json_file)
    class_dict = {  'hat': 1,
                    'person': 2 }

    #  Read train.txt All line information in , Delete blank lines 
    with open(train_txt_path, "r") as r:
        train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc Information transfer yolo, And copy the image file to the corresponding folder 
    translate_info(train_file_names, save_file_root, class_dict, "train")

    #  Read val.txt All line information in , Delete blank lines 
    with open(val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc Information transfer yolo, And copy the image file to the corresponding folder 
    translate_info(val_file_names, save_file_root, class_dict, "val")


if __name__ == "__main__":
    main()