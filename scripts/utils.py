import xml.etree.ElementTree
import numpy
import json
import pathlib

def xml_to_json(filename):
    basename = pathlib.Path(filename).stem
    tree = xml.etree.ElementTree.parse(filename)
    root = tree.getroot()
    comments = []
    tag_ops  = {
        'review_id'  : lambda x: str(x),
        'name'       : lambda x: str(x),
        'user_id'    : lambda x: str(x),
        'commentaire': lambda x: str(x),
        'movie'      : lambda x: numpy.int(x),
        'note'       : lambda x: numpy.float(x.replace(',', '.')),
    }
    for child in root:
        comment  =  { tag: op(child.find(tag).text) for tag, op in tag_ops.items() }
        comments += [ comment ]
    with open(f'../data/json/{basename}.json', 'w', encoding='utf8') as file:
        json.dump(comments, file, ensure_ascii=False)