import os
import yaml

import pandas as pd
import xml.etree.ElementTree as ET

from types import SimpleNamespace
from sklearn.model_selection import train_test_split

from utils.experiment_utils import create_linspace
from utils.preprocess import *


SOURCE_PATH = './source_data'
DATA_PATH = './data'
CONFIG_PATH = './conf'
DATASETS = ['ami', 'emoevent', 'haternet', 'hateval2019', 'mex-a3t', 'universal_joy', 'tass2019', 'detoxis']


class Colors:
    BLACK = '\033[1;30m'
    RED = '\033[1;31m'
    GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[1;34m'
    PURPLE = '\033[1;35m'
    CYAN = '\033[1;36m'
    WHITE = '\033[1;37m'
    ENDC = '\033[0m'


def colored(text, color):
    return f'{color}{text}{Colors.ENDC}'


def write_split_files(dataset, trn, dev, tst):
    trn.to_csv(os.path.join(DATA_PATH, dataset, 'train_es.tsv'), index=False, sep='\t', mode='w')
    dev.to_csv(os.path.join(DATA_PATH, dataset, 'dev_es.tsv'), index=False, sep='\t', mode='w')
    tst.to_csv(os.path.join(DATA_PATH, dataset, 'test_es.tsv'), index=False, sep='\t', mode='w')


def prepare_files():
    seed = 100
    test_ratio = 0.2

    # EmoEvent and HaterNet
    filename = 'original_es.tsv'
    data = {'emoevent': pd.read_csv(os.path.join(SOURCE_PATH, 'emoevent', filename), sep='\t'),
            'haternet': pd.read_csv(os.path.join(SOURCE_PATH, 'haternet', filename), sep=';\\|\\|;',
                                    names=['id', 'text', 'hateful'],
                                    header=None,
                                    engine="python")}
    labels = {'emoevent': 'offensive',
              'haternet': 'hateful'}

    for dataset in data:
        data[dataset].text = basic_text_normalization(data[dataset].text)
        y = data[dataset][labels[dataset]]
        trn, tst = train_test_split(data[dataset], shuffle=True, test_size=test_ratio, stratify=y, random_state=seed)
        y = trn[labels[dataset]]
        trn, dev = train_test_split(trn, shuffle=True, test_size=test_ratio, stratify=y, random_state=seed)
        write_split_files(dataset, trn, dev, tst)
        print(f'Dataset: {dataset} --> N. Instances: {data[dataset].shape[0]} --> Train, Dev., Test: '
              f'{trn.shape[0]}, {dev.shape[0]}, {tst.shape[0]}')

    # HatEval 2019
    dataset = 'hateval2019'
    n_instances = {}

    for phase in ['train', 'dev', 'test']:
        data = pd.read_csv(os.path.join(SOURCE_PATH, dataset, f'original_{phase}_es.csv'), sep=',')
        data.text = basic_text_normalization(data.text)
        data.to_csv(os.path.join(DATA_PATH, dataset, f'{phase}_es.tsv'), index=False, sep='\t', mode='w')
        n_instances[phase] = data.shape[0]

    print(f'Dataset: {dataset} --> N. Instances: {sum(n_instances.values())} --> Train, Dev., Test: '
          f'{n_instances["train"]}, {n_instances["dev"]}, {n_instances["test"]}')

    # MEX-A3T
    dataset = 'mex-a3t'
    columns = ['text', 'aggressiveness']
    trn = pd.read_csv(os.path.join(SOURCE_PATH, dataset, 'original_train.tsv'), sep='\t', names=columns)
    tst = pd.read_csv(os.path.join(SOURCE_PATH, dataset, 'original_test.tsv'), sep='\t', names=columns)

    trn, dev = train_test_split(trn, shuffle=True, test_size=test_ratio, stratify=trn.aggressiveness, random_state=seed)
    for subset in [trn, dev, tst]:
        subset.text = basic_text_normalization(subset.text)
    write_split_files(dataset, trn, dev, tst)
    print(f'Dataset: {dataset} --> N. Instances: {trn.shape[0] + dev.shape[0] + tst.shape[0]} --> Train, Dev., Test: '
          f'{trn.shape[0]}, {dev.shape[0]}, {tst.shape[0]}')

    # TASS 2019
    dataset = 'tass2019'
    n_instances = {}
    for phase in ['train', 'dev', 'test']:
        phase_data = pd.DataFrame()
        for country in ['ES', 'CR', 'MX', 'PE', 'UY']:
            root = ET.parse(os.path.join(SOURCE_PATH, dataset, f'TASS2019_country_{country}_{phase}.xml')).getroot()
            tweets = []
            for item in root.iter('tweet'):
                tweet = {'country': country}
                for tweet_field in item.iter():
                    if tweet_field.tag not in ['tweet', 'sentiment', 'polarity']:
                        tweet[tweet_field.tag] = tweet_field.text
                tweets.append(tweet)
            phase_data = phase_data.append(tweets)
        new_cols = {'tweetid': 'tweet_id', 'content': 'text', 'user': 'user_id', 'value': 'polarity'}
        phase_data.rename(columns=new_cols, inplace=True)
        phase_data = phase_data[['tweet_id', 'user_id', 'country', 'date', 'text', 'polarity']]
        phase_data.text = basic_text_normalization(phase_data.text)
        phase_data.to_csv(os.path.join(DATA_PATH, dataset, f'{phase}_es.tsv'), index=False, sep='\t', mode='w')
        n_instances[phase] = phase_data.shape[0]

    print(f'Dataset: {dataset} --> N. Instances: {sum(n_instances.values())} --> Train, Dev., Test: '
          f'{n_instances["train"]}, {n_instances["dev"]}, {n_instances["test"]}')

    # Universal Joy
    dataset = 'universal_joy'
    trn_data = {}
    for filename in ['small', 'large', 'combi']:
        trn_data[filename] = pd.read_csv(os.path.join(SOURCE_PATH, dataset, filename + '.csv'))
        trn_data[filename] = trn_data[filename][trn_data[filename].language == 'es']
        trn_data[filename].text = trn_data[filename].text.apply(universal_joy_cleaning)

    # Apparently, spanish comments in 'large' and 'combi' are the same and 'small' is created using a subset of those
    trn = pd.concat(trn_data.values(), axis=0, ignore_index=True)
    trn.drop_duplicates(inplace=True, subset='text')

    # There is no overlapping between training, validation and test (also, they do not contain duplicates)
    dev = pd.read_csv(os.path.join(SOURCE_PATH, dataset, 'val.csv'))
    dev.drop_duplicates(inplace=True, subset='text')
    tst = pd.read_csv(os.path.join(SOURCE_PATH, dataset, 'test.csv'))
    tst.drop_duplicates(inplace=True, subset='text')
    # The test set approximately represents 12.5% of the total data
    # print(tst.shape[0]/(trn.shape[0] + dev.shape[0] + tst.shape[0]))

    # DETOXIS
    dataset = 'detoxis'

    trn = pd.read_csv(os.path.join(SOURCE_PATH, dataset, f'train.csv'), sep=',')
    tst = pd.read_csv(os.path.join(SOURCE_PATH, dataset, f'test.csv'), sep=',')

    trn, dev = train_test_split(trn, shuffle=True, test_size=test_ratio, stratify=trn.toxicity_level, random_state=seed)
    for subset in [trn, dev, tst]:
        subset.rename(columns={'comment': 'text'}, inplace=True)
        subset.text = basic_text_normalization(subset.text)
    write_split_files(dataset, trn, dev, tst)
    print(f'Dataset: {dataset} --> N. Instances: {trn.shape[0] + dev.shape[0] + tst.shape[0]} --> Train, Dev., Test: '
          f'{trn.shape[0]}, {dev.shape[0]}, {tst.shape[0]}')


def read_datasets(datasets, tasks, lang='es'):
    data = {}
    for dataset in datasets:
        if dataset not in DATASETS:
            raise Exception(f'Dataset {dataset} is not in the list of available datasets!')

        data[dataset] = {
            'trn': pd.read_csv(os.path.join(DATA_PATH, dataset, f'train_{lang}.tsv'), sep='\t'),
            'dev': pd.read_csv(os.path.join(DATA_PATH, dataset, f'dev_{lang}.tsv'), sep='\t'),
            'tst': pd.read_csv(os.path.join(DATA_PATH, dataset, f'test_{lang}.tsv'), sep='\t')
        }

        for phase in data[dataset]:
            data[dataset][phase] = data[dataset][phase][['text'] + tasks[dataset]]

    return data


def create_namespace_from_dict(dic, name=None):
    for k, v in dic.items():
        if isinstance(v, dict):
            dic[k] = create_namespace_from_dict(v, k)
    ns = SimpleNamespace(**dic)
    ns.__name__ = name
    return ns


def process_config(dic, name=None):
    for k, v in dic.items():
        if k not in ['transfer_learning', 'optimization']:
            if isinstance(v, dict):
                dic[k] = process_config(v, k)
            elif isinstance(v, list):
                for vi in v:
                    if isinstance(vi, dict):
                        dic[k] += create_linspace(vi)
                        dic[k] = dic[k][1:]
            else:
                dic[k] = [v]
    return dic


def load_config(config_file):
    with open(os.path.join(CONFIG_PATH, config_file), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return process_config(config)  # create_namespace_from_dict(config)


def log(string, indent=0):
    start = '\t' * indent
    print(f'{start}{string}')
