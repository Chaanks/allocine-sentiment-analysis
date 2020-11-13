import xml.etree.ElementTree
import numpy
import json
import pathlib
import re

from pathlib import Path, PurePath


def xml_to_json(filename: Path, out: Path):
    """
    Convert an `.xml` file to a `.json` file given the following tree structure:
        <comments>
            <comment>
                <movie>       {str}   </movie>
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
    filename: Path
        Path to `.xml` file to parse
    out: Path
        Path to `.json` file to write
    """
    tree     = xml.etree.ElementTree.parse(filename)
    root     = tree.getroot()
    comments = []
    tag_ops  = {
        'review_id'  : lambda x: str(x),
        'name'       : lambda x: str(x),
        'user_id'    : lambda x: str(x),
        'commentaire': lambda x: str(x),
        'movie'      : lambda x: str(x),
        'note'       : lambda x: numpy.float(x.replace(',', '.')),
    }
    
    for child in root:
        comment   = { tag: op(child.find(tag).text) for tag, op in tag_ops.items() }
        comments += [ comment ]
    
    with open(out, 'w', encoding='utf8') as file:
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
    re_punctuation = re.compile(r"[^(a-z)|#|@|è|é|à|ù|ü|ë|ä|û|î|ê|â\s]")
    re_hyperlink   = re.compile(r"http\S+")
    re_extra_space = re.compile(r"\s+")
    
    tmp            = re_hyperlink.sub(' ', comment.lower())
    tmp            = re_punctuation.sub(' ', tmp)
    tmp            = re_extra_space.sub(' ', tmp)
    
    return [ { 'text': word } for word in tmp.split() ]


def json_to_ndjson(filename: Path, out_folder: Path, limit: int=100_000):
    """
    Get the raw list of comments form a `.json` source, tokenize the 
    `commentaire` section and register the generated list into a `.ndjson` 
    file, under the `ndjson`. This function splits the data into multiple 
    files if the number of `comment` objects exceeds a given limit. 
    
    Parameters
    ----------
    filename  : Path
        Original`.json` file to parse with untokenized comment.
    out_folder: Path
        Elastic search folder to register the list of tokenized comments.
    limit     : int, optional (default: 200_000)
        Maximum number of comments per file in the output pipeline.
    """
    basename = filename.stem
    # Creates folder given file's basename where all output files should be 
    # saved an update the output path.
    pathlib.Path(out_folder / f'{basename}').mkdir(parents=True, exist_ok=True)
    out_folder = out_folder / f'{basename}'
    
    # Loading raw json file 
    with open(filename, 'r', encoding='utf8') as file:   
        raw_dataset = json.load(file)
        
    # Tokenization and comments registration
    num_part   = 0
    num_elts   = len(raw_dataset)
    num_digits = len(str(num_elts))
    
    # Clears first file to edit, in case of previous data registration
    open(out_folder / f'{basename}_{num_part}.ndjson', "w").close()
    
    for idx, comment in enumerate(raw_dataset):
        comment['lst_mots'] = tokenize_comment(comment['commentaire'])
                
        if idx % limit == limit - 1:
            # The maximum number of entries has been reached, we need to 
            # register the next comments to a new file
            num_part   += 1
            open(out_folder / f'{basename}_{num_part}.ndjson', "w").close()
        
        # Registers data into the current `.ndjson` output file
        with open(out_folder / f'{basename}_{num_part}.ndjson', "a+") as file:
            file.write(f'{json.dumps(comment, ensure_ascii=False)}\n')

        print(f'progress: {idx:>{num_digits}_d} / {num_elts:_d}', end='\r')

    print(f'Done! [ num_items: {num_elts:_d} ]')
        
