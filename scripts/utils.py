import xml.etree.ElementTree
import numpy
import json
import pathlib
import re

def xml_to_json(filename: str):
    """
    Convert an `.xml` file to a `.json` file given the following tree structure:
    <comments>
        <comment>
            <movie>       {int}   </movie>
            <review_id>   {str}   </review_id>
            <name>        {str}   </name>
            <user_id>     {str}   </user_id>
            <note>        {float} </note>
            <commentaire> {str}   </commentaire>
        </comment>
        ...
     </comments>
    
    Parameter
    ---------
    filename: str
        Path to `.xml` file to parse
    """
    basename = pathlib.Path(filename).stem
    tree     = xml.etree.ElementTree.parse(filename)
    root     = tree.getroot()
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

def tokenize_comment(comment: str) -> list:
    """
    Tokenize a `comment` by removing hyperlinks, punctuation and extra spaces.
    
    Parameter
    ---------
    comment: str
        String to tokenize
    
    Returns
    -------
    list(str)
        The list of tokens extracted from the original comment
    """
    re_punctuation = re.compile(r"http\S+|[^(a-z)|#|@|è|é|à|ù|ü|ë|ä|û|î|ê|â\s]")
    re_hyperlink   = re.compile(r"http\S+")
    re_extra_space = re.compile(r"\s+")
    tmp            = re_hyperlink.sub(' ', comment.lower())
    tmp            = re_punctuation.sub(' ', tmp)
    tmp            = re_extra_space.sub(' ', tmp)
    return [ { 'text': word } for word in tmp.split() ]
    