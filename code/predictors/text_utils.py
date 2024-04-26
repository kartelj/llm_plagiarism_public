import json
from spacy.lang.sr import Serbian
from srtools import latin_to_cyrillic, cyrillic_to_latin
from sklearn.feature_extraction.text import CountVectorizer 
from scipy.sparse import lil_matrix
from unicodedata import normalize, category

def lemmatize(texts, lemma_path):
    print(f'Applying lemma rules from {lemma_path}...')
    with open(lemma_path, 'r', encoding='utf-8') as f:
        lemmas = json.load(f)
    nlp = Serbian()
    tokenizer = nlp.tokenizer
    new_texts = []
    for text in texts:
        words = tokenizer(text)
        new_text = ''
        for word in words:
            new_word = word.text
            cyrillic = latin_to_cyrillic(new_word)
            if cyrillic in lemmas:  
                new_word =lemmas[cyrillic]
            new_word = cyrillic_to_latin(new_word)
            if word.is_punct:
                new_text+=new_word
            else:
                new_text+=(' '+new_word)
        new_texts.append(new_text)
    return new_texts

def strip_accents(s):
   return ''.join(c for c in normalize('NFD', s) if category(c) != 'Mn')

def preprocess_text(texts, lemma_path):
    print('Normalizing texts...')
    for i in range(len(texts)):
        texts[i] = texts[i].lower() #strip_accents(texts[i]).lower()
    # lemmatization
    texts = lemmatize(texts, lemma_path)
    return texts

def extract_char_ngrams(texts, lemma_path, char_ngrams_size):
    texts = preprocess_text(texts, lemma_path)
    print('Extracting character n-grams...')
    ngram_char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,char_ngrams_size))
    X_ngram_char= ngram_char_vectorizer.fit_transform(texts)
    feature_names = ngram_char_vectorizer.get_feature_names_out()
    return X_ngram_char, feature_names
    
def extract_word_ngrams(texts, lemma_path, word_ngrams_size):
    texts = preprocess_text(texts, lemma_path)
    # word features
    print('Extracting word n-grams...')
    ngram_word_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,word_ngrams_size))
    X_ngram_word = ngram_word_vectorizer.fit_transform(texts)
    feature_names = ngram_word_vectorizer.get_feature_names_out()
    return X_ngram_word, feature_names

def transform_to_matrix(texts, features, lemma_path, ngram_type, ngrams_max_size):
    # now create feature vector dicts that are compatible with model features, if feature is not in there, then just ignore it
    if ngram_type=='char':
        X_tmp, tmp_features = extract_char_ngrams(texts,lemma_path, ngrams_max_size)
    elif ngram_type=='word':
        X_tmp, tmp_features = extract_word_ngrams(texts, lemma_path, ngrams_max_size)
    else:
        raise Exception(f'Not recognized ngram_type={ngram_type}.')
    X = __adjust_features(X_tmp, tmp_features, features)
    return X

def __adjust_features(X_tmp, features_tmp, features):
    # adjusting to model features and creating vectors of features of the same dimension
    features_dict = {}
    for i in range(len(features)):
        features_dict[features[i]] = i
    X_test = lil_matrix((X_tmp.shape[0], len(features)))
    for i, j in zip(*X_tmp.nonzero()):
        if features_tmp[j] in features_dict:
            X_test[i, features_dict[features_tmp[j]]] = X_tmp[i, j]
    return X_test.tocsr()