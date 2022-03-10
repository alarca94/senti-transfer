import re
import json

# from ekphrasis.classes.preprocessor import TextPreProcessor


def translate_emojis(posts, lang='es'):
    with open('./data/simple_emojis.json', 'r') as f:
        simple_emojis = json.load(f)
    with open('./data/complex_emojis.json', 'r') as f:
        complex_emojis = json.load(f)

    # print('##### Replacing simple emojis...')
    pattern = '|'.join(sorted(re.escape(k) for k in simple_emojis))
    posts = posts.apply(lambda t: re.sub(pattern,
                                         lambda m: simple_emojis.get(m.group(0).upper()).get(lang),
                                         t,
                                         flags=re.IGNORECASE))

    # print('##### Replacing complex emojis...')
    pattern = '|'.join(sorted(re.escape(k) for k in complex_emojis))
    return posts.apply(lambda t: re.sub(pattern,
                                        lambda m: complex_emojis.get(m.group(0).upper()).get(lang),
                                        t,
                                        flags=re.IGNORECASE))


def normalize_laughs(posts):
    # print('##### Normalizing laughs...')
    return posts.apply(lambda t: re.sub('[jha]{5,}', 'jajaja', t))


def handle_url(posts, mode='mask'):
    # print('##### Masking URLs...')
    url_regex = '(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
    replace_txt = ''
    if mode == 'mask':
        replace_txt = 'URL'
    return posts.apply(lambda t: re.sub(url_regex, replace_txt, t))


def handle_user(posts, mode='mask'):
    # print('##### Masking users...')
    replace_txt = ''
    if mode == 'mask':
        replace_txt = 'USUARIO'
    return posts.apply(lambda t: re.sub('@[\w]+', replace_txt, t))


def handle_hashtag(posts, mode='mask'):
    # print('##### Masking hashtags...')
    replace_txt = ''
    if mode == 'mask':
        replace_txt = 'HASHTAG'
    return posts.apply(lambda t: re.sub('#[\w]+', replace_txt, t))


def basic_text_normalization(data):
    data = handle_hashtag(data)
    data = handle_user(data)
    data = handle_url(data)
    return data


def universal_joy_cleaning(text):
    text = re.sub('\[PHOTO]', 'FOTO', text)
    text = re.sub('\[PERSON]', 'PERSONA', text)
    text = re.sub('\[WITH]', 'con PERSONA', text)
    text = re.sub('\s+at\s+\[LOCATION]', ' en LUGAR', text)
    text = re.sub('\[LOCATION]', 'LUGAR', text)
    text = re.sub('\[URL]', 'URL', text)
    text = re.sub('\[EMAIL]', 'EMAIL', text)
    return text


def run_config(config, data):
    """
    :param config: Configuration dictionary e.g.
                    example = {
                        dates: mask
                        hashtags: segment
                        numbers: mask
                        users: mask
                        urls: mask
                    }
    :param data: it can either be a dictionary containing trn-dev-tst sets or a single set (pd.DataFrame)
    :return: preprocessed data according to the configuration
    """
    if isinstance(data, dict):
        new_data = []
        for key in ['trn', 'dev', 'tst']:
            new_data.append(run_config(config, data[key]))
        return new_data
    else:
        # TODO: Use config to preprocess the data
        tmp_data = data.copy()
        return tmp_data
