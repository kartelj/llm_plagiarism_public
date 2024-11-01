# test each serialized model for different temperatures, text input sizes and different sources 
import os
import subprocess
import pandas as pd

#with open('results.csv', 'w') as file_res:
#    file_res.write('model_name,language,source,domain,same_length,temperature,accuracy\n')

for lang_short, lang_long in {'en':'English', 'sr':'Serbian'}.items():
    model_directory_path = f'./models/{lang_short}/'
    for model_name in os.listdir(model_directory_path):
        print(f'Doing model {model_name}')
        model_path = os.path.join(model_directory_path, model_name)
        if os.path.isfile(model_path):
            # apply each model to each test variant
            for source in ['Singi', 'UB']:
                for domain in ['Computers', 'Marketing']:
                    for temp in [0, 0.5, 1]:
                        for same_len in [True, False]:
                            setting = f'{model_name},{lang_long},{source},{domain},{same_len},{temp}'
                            with open('results.csv', 'r') as file_res:
                                content = file_res.read()
                                if setting in content:
                                    print(f'Skipping {setting}')
                                    continue
                            test_data_path = f'../data/ParameterAnalysis/{source}/{domain}/{lang_long}/T{temp}{"_SameLen" if same_len else ""}'
                            if os.path.exists('tmp.csv'):
                                os.remove('tmp.csv')
                            command = ['python', '../code/predictors/test.py', test_data_path, f'../code/predictors/{lang_short}_lemma_lookup.json', model_path, 'tmp.csv']
                            print(command)
                            result = subprocess.run(command, capture_output=True, text=True)
                            #print(f'Output: {result.stdout} {result.stderr}')
                            try:
                                df = pd.read_csv('tmp.csv')
                                correct = 0
                                for index, row in df.iterrows():
                                    if 'G-' in df.iloc[index, 0]:
                                        if df.iloc[index,1]==0:
                                            correct+=1
                                    else:
                                        if df.iloc[index,1]==1:
                                            correct+=1
                                result_string = f'{setting},{correct/len(df)}'
                                print(result_string)
                                with open('results.csv', 'a') as file_res:
                                    file_res.write(result_string+'\n')
                            except Exception as ex:
                                print(ex)