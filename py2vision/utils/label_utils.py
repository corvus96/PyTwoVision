import os

def label_map(labels, dst_path=None, name='label_map'):
    """An easy way to convert classes names and ids to a .pbtxt file compatible with tensorflow models API.
    
    Args:
        labels: a list, which each element is a dictionary with two keys 'name' and 'id'.
        dst_path: a path where the file will be saved.
        name: the name of the file, with which it will save the .pbtxt

    Returns:
        a string where the file was saved.

    Raises:
        Exception: When someone element in internal dictionaries have another keys different of name and id.
        TypeError: when labels aren't list type
        ValueError: when labels are empty
    """
    if type(labels) != list: raise TypeError('labels should be a list of dictionaries!')
    if not labels: raise ValueError('labels are empty')
    if dst_path == None:
        dst_path = os.getcwd()
    file_path = dst_path +  '/{}.pbtxt'.format(name)
    allowed_keys = ['name', 'id']
    with open(file_path, 'w') as f:
        for label in labels:
            if allowed_keys == list(label.keys()):
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')
            else: 
                os.remove(file_path)
                raise Exception("Your keys should be only 'name' and 'id'")
        print('file saved in: {}'.format(file_path))
    
    return file_path
            
def index2class(index, classes: list):
    """Convert index (int) to class name (string)"""
    return classes[index]


def class2index(class_name, classes: list):
    """Convert class name (string) to index (int)"""
    return classes.index(class_name)

def class_text_to_int(row_label, label_map_dict: dict):
    return label_map_dict[row_label]

def read_class_names(class_file_name):
    """loads class name from a file to a dict"""
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names