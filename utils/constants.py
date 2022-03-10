import numpy as np

CAT_TYPE = 'categorical'
ORD_TYPE = 'ordinal'
BIN_TYPE = 'binary'

DATASET_TASKS = {
    'ami': [
        'misoginous',
        'misogyny_category',
        'target'
    ],
    'emoevent': [
        ('emotion', CAT_TYPE),
        'offensive'
    ],
    'hateval2019': [
        'HS',
        'TR',
        'AG'
    ],
    'detoxis': [
        'argumentation',
        'constructiveness',
        'positive_stance',
        'negative_stance',
        'sarcasm',
        'target_person',
        'target_group',
        'stereotype',
        'mockery',
        'insult',
        'improper_language',
        'intolerance',
        'aggressiveness',
        'toxicity',
        ('toxicity_level', ORD_TYPE)
    ],
    'haternet': ['hateful'],
    'mex-a3t': ['aggressiveness'],
    'universal_joy': [('emotion', CAT_TYPE)],
    'tass2019': [('polarity', ORD_TYPE, {'N': 0, 'NEU': 1, 'P': 2})],
}

SEED = np.random.RandomState(42)
