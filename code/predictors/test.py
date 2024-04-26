from model_descriptor import ModelDescriptor
from typing import List
from sys import argv
from os import walk

if __name__ == "__main__":
    if len(argv)!=5:
        print(f'Usage: <dataset_path> <lemma_path> <model_path> <output_path>')
        exit(1)
    dataset_path = argv[1] 
    lemma_path = argv[2]
    model_path = argv[3]
    output_path = argv[4] 
    
    texts = []
    file_paths = []
    for parent, _, files in walk(dataset_path):
        for file in files:
            try:
                full_path = parent+'/'+file
                with open(full_path, mode='r', encoding='utf-8') as f:
                    text = f.read()
                    texts.append(text)
                
                file_paths.append(file)
            except Exception as ex:
                print(ex)

    # now applying all models that are consistent with model_filter_mask
    print(f'Applying model {model_path}.')
    md = ModelDescriptor.load_from_file(model_path)
    yp = md.predict(texts, lemma_path)
    with open(output_path, 'w') as f: 
        for i in range(len(yp)):
            f.write(f'{file_paths[i]},{yp[i]}\n')
    print(f'Results saved to {output_path}.')
