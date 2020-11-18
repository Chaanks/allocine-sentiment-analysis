import xml.etree.ElementTree
import numpy
import json
import pathlib
import re
import shutil
import math

from pathlib import Path, PurePath


def xml_to_json(xml_file: Path, json_file: Path):
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
    xml_file : Path
        Path to `.xml` file to parse
    json_file: Path
        Path to `.json` file to write
    """
    tree     = xml.etree.ElementTree.parse(xml_file)
    root     = tree.getroot()
    comments = []
    tag_ops  = {
        'review_id'  : lambda x: str(x),
        'name'       : lambda x: str(x),
        'user_id'    : lambda x: str(x),
        'commentaire': lambda x: str(x),
        'movie'      : lambda x: { 'id': str(x) },
        'note'       : lambda x: numpy.float(x.replace(',', '.')),
    }
    
    for child in root:
        comment   = { tag: op(child.find(tag).text) for tag, op in tag_ops.items() }
        comments += [ comment ]
    
    with open(json_file, 'w', encoding='utf8') as file:
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


def json_to_ndjson(
    json_file       : Path, 
    out_folder      : Path, 
    limit           : int  = 200_000, 
    es_index        : str  = 'movie_db',
    movie_additional: Path = None
):
    """
    Get the raw list of comments form a `.json` source, tokenize the 
    `commentaire` section and register the generated list into a `.ndjson` 
    file, under the `ndjson`. This function splits the data into multiple 
    files if the number of `comment` objects exceeds a given limit.
    
    Parameters
    ----------
    json_file       : Path
        Original `.json` file to parse with untokenized comment.
    movie_additional: Path, optional (default: None)
        Path to additional data on movies as a `.json`file. This aims at adding 
        some extra information on movie reviews from the raw `.xml` dataset.
    out_folder      : Path
        Elastic search folder to register the list of tokenized comments.
    limit           : int, optional (default: 200_000)
        Maximum number of comments per file in the output pipeline.
    es_index        : str, optional (default: "movie_db")
        Name of the Elastic Search database.
    """
    basename = json_file.stem
    
    # Creates folder given file's basename where all output files should be 
    # saved an update the output path.
    pathlib.Path(out_folder / f'{basename}').mkdir(parents=True, exist_ok=True)
    out_folder = out_folder / f'{basename}'
    
    if out_folder.is_dir():
        print(f'\033[0;37mCleaning \033[0;37m{out_folder} \033[0;37m..\033[0m', end=' ')
        clear_dir(out_folder)
        print(f'\033[0;34mDone!\033[0m')
    
    es_idx = { "index": { "_index": es_index } }

    # Loading raw json file
    with open(json_file, 'r', encoding='utf8') as file:   
        raw_dataset = json.load(file)
    
    # Loading complementary data from web scrapping
    if movie_additional is not None:
        with open(movie_additional, 'r', encoding='utf8') as file:
            extra_info = json.load(file)

    # Tokenization and comments registration
    num_part   = 0
    num_elts   = len(raw_dataset)
    num_digits = len(f'{num_elts:_d}')
    
    # Clears first file to edit, in case of previous data registration
    open(out_folder / f'{basename}_{num_part}.ndjson', "w").close()
    
    for idx, review in enumerate(raw_dataset):
        review['lst_mots']       = tokenize_comment(review['commentaire'])
        review['num_chars']      = sum(len(token) for token in review['lst_mots'])
        
        if extra_info is not None:
            movie_id = review['movie']['id']
            
            review['movie']['title']    = extra_info[movie_id]['title']
            
            review['movie']['category'] = extra_info[movie_id]['category']
            
            avg_note = None
            is_users = False
            for content in extra_info[movie_id]['data']:
                if content.strip().lower() == 'spectateurs':
                    is_users = True
                else:
                    try:
                        content = float(content.strip().replace(',', '.'))
                        if not math.isnan(content) and is_users:
                            review['movie']['avg_note'] = content
                            break
                    except ValueError:
                        continue
             
        
                        
        if idx % limit == limit - 1:
            # The maximum number of entries has been reached, we need to 
            # register the next comments to a new file
            num_part   += 1
            open(out_folder / f'{basename}_{num_part}.ndjson', "w").close()
        
        # Registers data into the current `.ndjson` output file
        with open(out_folder / f'{basename}_{num_part}.ndjson', "a+") as file:
            file.write(f'{json.dumps(es_idx, ensure_ascii=False)}\n')
            file.write(f'{json.dumps(review, ensure_ascii=False)}\n')

        print(f'\033[0;37mProgress: \033[1;30m{idx:>{num_digits}_d}\033[0m /{num_elts:_d}', end='\r')

    print(f'\033[0;34mDone!\033[0m [ \033[0;37mnum_items: {num_elts:_d}\033[0m ]')


def clear_dir(folder: Path):
    for path in folder.glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def xml_to_ndjson(
    xml_file        : Path, 
    json_folder     : Path, 
    es_folder       : Path, 
    movie_additional: Path = None
):
    """
    Parses a raw xml file containing the list of all movie reviews and convert 
    it to a `.ndjson` file, with  extra information on reviews if additional
    data are available.
    
    Parameters
    ----------
    xml_file        : Path
        Path to the `.xml` file to parse.
    json_folder     : Path
        Path to the `.json` folder that should contain the intermediate parsing
        results.
    es_folder       : Path
        Path to the elastic search folder that should contain the parsed result
    movie_additional: Path
        Path to the file containing the extra data on movies shown in the 
        dataset.
    """
    basename = xml_file.stem
    
    # Name of the intermediate `.json` file
    json_file = json_folder / f'{basename}.json'
    
    print(f'\033[1;33mParsing \033[1;37m{xml_file}\033[1;33m file..\033[0m', end=' ')
    xml_to_json(xml_file,  json_file)
    print(f'\033[1;34mDone!\033[0m')
    
    # Parse raw `.json` movie reviews (i.e. directly extracted from the `.xml` 
    # movie reviews) file into a `.ndjson` file.
    print(f'\033[1;33mGenerating corresponding .ndjson file to \033[1;37m{es_folder}\033[1;33m ..\033[0m')
    json_to_ndjson(json_file, es_folder, movie_additional = movie_additional, limit = 10_000)
