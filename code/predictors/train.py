# Tests various predictors for checking if text is GPT or Human created
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import BernoulliNB
from sklearn.semi_supervised import LabelSpreading
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from os import walk
from time import time
from random import seed, random
from model_descriptor import ModelDescriptor
from text_utils import extract_char_ngrams, extract_word_ngrams
from sys import argv, exit
from sklearn.metrics import accuracy_score
from text_utils import transform_to_matrix

def get_classifiers_and_param_grids():
    classifiers = {
        "AdaBoostClassifier": (AdaBoostClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]}),
        "BaggingClassifier": (BaggingClassifier(), { 'n_estimators': [10, 50, 100], 'max_samples': [0.5, 1.0], 'max_features': [0.5, 1.0]}), 
        "BernoulliNB": (BernoulliNB(), {"alpha": [0.1, 0.5, 1.0]}),
        "CalibratedClassifierCV": ((CalibratedClassifierCV(), None)),
        "DecisionTreeClassifier": (DecisionTreeClassifier(), {"max_depth": [None, 5, 10, 20]}),
        "GradientBoostingClassifier": (GradientBoostingClassifier(), {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]}),
        "ExtraTreesClassifier": (ExtraTreesClassifier(), {'max_depth': [None, 5],'min_samples_split': [2, 5],'min_samples_leaf': [1, 2],'bootstrap': [True, False]}),
        "LabelSpreading": (LabelSpreading(), {"kernel": ['knn', 'rbf']}),
        "LinearSVC": (LinearSVC(), {"C": [0.1, 1.0, 10.0]}),
        "LogisticRegression": (LogisticRegression(), {"C": [0.1, 1.0, 10.0]}),
        "MLPClassifier": (MLPClassifier(), {"alpha": [0.0001, 0.001, 0.01]}),
        "NuSVC": (NuSVC(), {"nu": [0.1, 0.5, 1.0]}),
        "PassiveAggressiveClassifier": (PassiveAggressiveClassifier(), {"C": [0.1, 1.0, 10.0]}),
        "Perceptron": (Perceptron(), {"alpha": [0.0001, 0.001, 0.01]}),
        "RandomForestClassifier": (RandomForestClassifier(), {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]}),
        "RidgeClassifierCV": (RidgeClassifierCV(), None),
        "SGDClassifier": (SGDClassifier(), {"alpha": [0.0001, 0.001, 0.01]}),
        "SVC": (SVC(), {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "poly", "rbf", "sigmoid"]}),
        "XGBClassifier": (XGBClassifier(), {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]})
    }
    #classifiers = {"XGBClassifier": (XGBClassifier(), None)}
    return classifiers

def read_all_files(path):
    files_info = {}
    domains = ['arh', 'biof', 'ekf', 'etf', 'farmf', 'fhf', 'fillf', 'filzf', 'fizf', 'geof', 'grf', 'hemf', 'maf', 'medf', 'onf', 
               'pbf', 'pmf', 'pnf', 'poljf', 'prf', 'rgf', 'serf', 'sf', 'sfvf', 'stomf', 'sumf', 'tbf', 'tmf', 'ucf', 'vmf']
    tmp = set([])
    for parent, _, files in walk(path):
        for file in files:
            full_path = parent+'/'+file
            try:
                with open(full_path, mode='r', encoding='utf-8') as f:
                    text = f.read()
                domain = None
                for d in domains:
                    if d+'-' in full_path:
                        domain = d
                        break
                if domain is None:
                    raise Exception(f'Unrecognized domain for path {full_path}.')
                tmp.add(domain)
                file_info = (text, 'Native' in parent, domain)
                files_info[file] = file_info
            except Exception:
                print(full_path)
    print(tmp)
    return files_info

def scores_per_group(yp, y, groups):
    group_counts = {}
    assert(len(yp)==len(y))
    assert(len(y)==len(groups))
    for i in range(len(y)):
        g = groups[i]
        if g in group_counts:
            g_info = group_counts[g]
        else:
            g_info = (0, 0)
        if y[i]==yp[i]:
            g_info = (g_info[0]+1, g_info[1]+1)
        else:
            g_info = (g_info[0], g_info[1]+1)
        group_counts[g]=g_info
    for g in group_counts.keys():
        print(f'{g}:\t{group_counts[g][0]*1.0/group_counts[g][1]}\t{group_counts[g][0]}/{group_counts[g][1]}')

def train(classifer, X, y, param_grid = None):
    print(f'Training vectors shape {X.shape}...')
    start = time()
    try:
        if param_grid is None:
            model = classifer.fit(X, y)
        else:
            search = GridSearchCV(classifer, param_grid, cv=5, verbose=True)
            search.fit(X, y)
            print(f'Best hyperparameters {search.best_params_} with score {search.best_score_}')
            model = search.best_estimator_
    except Exception as ex:
        # probably it doesn't work with sparse matrix
        print(f'Exception {ex}')
    train_time = time() - start
    return model, train_time

def split(all_keys, random_seed, train_size):
    seed(random_seed)
     # splitting raw texts to training and testing part
    print('Splitting to training/test parts...')
    # first split so that for a given thesis, there is a GPT and Native version inside
    all_keys_to_index = {}
    i = 0
    for k in all_keys:
        all_keys_to_index[k]=i
        i+=1
    training_row_indices = []
    test_row_indices = []
    for i in range(len(all_keys_to_index)):            
        rv = random()
        if rv<train_size:
            selected_indices = training_row_indices
        else:
            selected_indices = test_row_indices    
        if i in training_row_indices or i in test_row_indices:
            continue
        # add both GPT and Native to training_keys
        key = all_keys[i] 
        if 'G-' in key:
            other_key = key.replace('G-', '')
        else:
            other_key = 'G-'+key
        if not other_key in all_keys:
            continue # not both are inside the corpus, so we do not add neither of two
        selected_indices.append(all_keys_to_index[key])
        selected_indices.append(all_keys_to_index[other_key])
    return training_row_indices, test_row_indices

def evaluate_on_prepared_matrix(models, X, y, out_file):
    with open(out_file,'w') as f:
        f.write('') # just initialize file, possibly delete previous content
    for m in models:
        yp = m.predict(X)
        acc = accuracy_score(y, yp)
        output = f'{str(m)},{acc}\n'
        print(output)
        with open(out_file, 'a') as f:
            f.write(output)   

def main(dataset_path, lemma_path, random_seed, train_size, ngram_type, ngrams_max_size):
    if 'Serbian' in dataset_path:
        tag = 'sr'
    elif 'English' in dataset_path:
        tag = 'en'
    else:
        raise Exception('Unsupported dataset path {dataset_path}.')
    
    # preparing features from input abstracts
    print('Loading dataset...')
    files_info = read_all_files(dataset_path)
    print(f'Loaded {len(files_info)} files.')
 
    all_keys = [k for k in files_info.keys()]
    all_texts = [p[0] for p in files_info.values()]
    y = [1 if p[1] else 0 for p in files_info.values()]
    d = [p[2] for p in files_info.values()] # domains

    training_row_indices, test_row_indices = split(all_keys, random_seed, train_size)
    train_texts = [all_texts[i] for i in training_row_indices]

    # now create feature vector dicts
    if ngram_type=='char':
        X_train, features = extract_char_ngrams(train_texts,lemma_path, ngrams_max_size)
    elif ngram_type=='word':
        X_train, features = extract_word_ngrams(train_texts, lemma_path, ngrams_max_size)
    else:
        raise Exception(f'Not recognized ngram_type={ngram_type}.')
    
    # target variables
    y_train = [y[i] for i in training_row_indices]

    # performing various classifier trainings
    model_descriptors = []
    classifiers_and_param_grids = get_classifiers_and_param_grids()
    for c_name in classifiers_and_param_grids.keys():
        try:
            c, param_grid = classifiers_and_param_grids[c_name]
            print(f'Training {c_name}...')
            c.random_state = random_seed
            model, train_time = train(c, X_train, y_train, param_grid)
            md = ModelDescriptor(model, features, ngram_type, ngrams_max_size, train_time)
            md.save_to_file(f'{tag}_{c_name}_{random_seed}_{ngram_type}_{ngrams_max_size}')
            model_descriptors.append(md)
        except Exception as ex:
            print(ex)

    # evaluate on training data
    evaluate_on_prepared_matrix([md.model for md in model_descriptors], X_train, y_train, f'train_{tag}_{train_size}_{ngram_type}_{ngrams_max_size}_{random_seed}.csv')
    
    # evaluate on test data
    y_test = [y[i] for i in test_row_indices]
    test_texts = [all_texts[i] for i in test_row_indices]
    X_test = transform_to_matrix(test_texts, features, lemma_path, ngram_type, ngrams_max_size)
    evaluate_on_prepared_matrix([md.model for md in model_descriptors], X_test, y_test, f'test_{tag}_{train_size}_{ngram_type}_{ngrams_max_size}_{random_seed}.csv')
    
if __name__ == "__main__":
    if len(argv)!=7:
        print(f'Usage: <dataset_path> <lemma_path> <random_seed> <train_size> <ngram_type> <ngrams_max_size>')
        exit(1)
    dataset_path = argv[1] 
    lemma_path = argv[2] 
    random_seed = int(argv[3])
    train_size = float(argv[4])
    ngram_type = argv[5]
    ngrams_max_size = int(argv[6])
    print(f'Running test with parameters dataset_path={dataset_path} lemma_path={lemma_path} random_seed={random_seed} train_size={train_size} ngram_type={ngram_type} ngrams_max_size={ngrams_max_size}')
    main(dataset_path, lemma_path, random_seed, train_size, ngram_type, ngrams_max_size)