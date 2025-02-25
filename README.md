# Film rating prediction
Groupe | Date | Sujet | UE | UCE | Encadrants
:---|:---|:---|:---|:---|:---
Jarod Duret, Jonathan Heno | 14/01/2021 | Prédiction de notes de films | Innovations et recherche pour la société du numérique | Application d'innovation | Richard Dufour, Vincent Labatut, Mickaël Rouvier


## Installation
### conda
The conda environment used for this project can be generated as follows:
```shell
$ conda env create -f environment.yml
```

### pip
You can also install project's dependencies with `pip` with the following command:
```shell
$ pip install -r requirements.txt
```
Also, make sure to install the appropriate **cudatoolkit** version for GPU usage (`cudatoolkit 11.0.221` used).


## Creation
Commands used for generating the python configuration:
```shell
$ conda env export > environment.yml
$ pip freeze > requirements.txt
```

## Usage
### Generating datasets
```shell
usage: gen.py [-h] -d DSFILE [DSFILE ...] -if {raw,json} -of
              {json,es,score,run} [-sw STOPWORDS] [-e EXTRA] [-ei ES_IDX]
              [-std]
              out

Generate a set of dataset files from an input source to another with specified
format. NOTE: The input files should all have the same extension (either 'xml'
or 'json')

positional arguments:
  out                   Path to output directory.

optional arguments:
  -h, --help            show this help message and exit
  -d DSFILE [DSFILE ...], --data DSFILE [DSFILE ...]
                        Path to dataset file(s).
  -if {raw,json}, --in_format {raw,json}
                        Source file format.
  -of {json,es,score,run}, --out_format {json,es,score,run}
                        Output file format.
  -sw STOPWORDS, --stopwords STOPWORDS
                        [ES][SCORE][RUN] Path to stop words file.
  -e EXTRA, --extra EXTRA
                        [ES] Path to movie metadata set.
  -ei ES_IDX, --es_idx ES_IDX
                        [ES] Name of the ES database.
  -std, --standardize   [ES][SCORE][RUN] Review content standardization.
```

#### JSON
```shell
$ python3 src/py/gen.py -if raw -d data/xml/train.xml -of json data/json
```

#### Creating set of dataset chunks for exporting on ElasticSearch
```shell
$ python3 src/py/gen.py -if json -d data/json/train.json -of es -sw data/json/stopwords.json -std data/es
```

#### Training dataset
To create the training set from raw `.xml` data, we can enter the following command:
```shell
$ python3 src/py/gen.py -if raw -d data/xml/train.xml data/xml/dev.xml -of run -sw data/json/stopwords.json -std data/keras
```

#### Trials
To create the test dataset from raw `.xml` file as a series of trials with a standardized and filtered review:
```shell
$ python3 src/py/gen.py -if raw -d data/xml/test.xml -of score -sw data/json/stopwords.json -std data/csv
```

### Training a model
```shell
usage: run.py [-h] -cfg CONFIG -t TRAIN -d DEV out

Trains a neural network on allocine database given a configuration file.

positional arguments:
  out                   Path to output directory.

optional arguments:
  -h, --help            show this help message and exit
  -cfg CONFIG, --config CONFIG
                        Path to model config file.
  -t TRAIN, --train TRAIN
                        Path to training set.
  -d DEV, --dev DEV     Path to validation set.
```

#### Example:
```shell
$ python3 src/py/run.py -cfg cfg/conv5p.yml -t data/keras/train -d data/keras/dev data/model/conv5p
```

### Evaluating a pre-trained model
```shell
usage: score.py [-h] -m MODEL -t TRIALS out

Scores a trained model on a set of trials.

positional arguments:
  out                   Path to output directory.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to directory of the trained model.
  -t TRIALS, --trials TRIALS
                        Path to trials.
```

```shell
$ python3 src/py/score.py -m data/model/conv5p -t data/csv/test.csv out/
```

## To Do
- [x] Code refactor
- [x] Documentation
- [x] Config file management
- [x] Data generation
- [x] CNN improvement (5 convolutions instead of 2)
  - [x] Functional testing
  - [x] Fix
  - [x] Implementation
- [ ] User embedding generation
- [ ] Director one hot encoding on all corpus (train, dev and test)
- [ ] Genre multi-encoding for all films (train, dev and test)
- [ ] DNN aggregating CNN outputs, director's and genre's encoding 

