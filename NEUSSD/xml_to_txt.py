import os
import xml.etree.ElementTree as ET

pd = os.path.abspath(os.path.dirname(__file__))

def convert_annotation(xml_file, classes):
    """ Convert XML annotation to YOLO format """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    yolo_data = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes or int(obj.find('difficult').text) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        bb = (b[0] + b[2]) / (2 * w), (b[1] + b[3]) / (2 * h), (b[2] - b[0]) / w, (b[3] - b[1]) / h
        yolo_data.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
    
    return yolo_data

def convert_annotations_directory(annotations_dir, output_dir, classes):
    """ Convert annotations for an entire directory """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, xml_file)
            yolo_data = convert_annotation(xml_path, classes)
            output_file = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))
            with open(output_file, 'w') as file:
                file.write("\n".join(yolo_data))

# Define your classes
classes = [
    'crazing', 
    'inclusion', 
    'patches',
    'pitted_surface',
    'rolled-in_scale',
    'scratches'
]

# Paths to your data
annotations_dir = f'{pd}\\Data\\valid_annotations'  # Change to your XML annotations directory
output_dir = f'{pd}\\Data\\labels\\val'      # Change to your desired output directory

convert_annotations_directory(annotations_dir, output_dir, classes)
