import argparse
import json
import pandas as pd
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_real', type=str, required=True)
    parser.add_argument('--path2save', type=str, required=True)
    parser.add_argument('--noise_level', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--datacentric_treshold', type=str, required=True)
    parser.add_argument('--stage', type=str, required=True)
    parser.add_argument('--model',type=str, required=False)
    args = parser.parse_args()
    
    # Desserializar as strings JSON em listas
    noise_level = json.loads(args.noise_level)
    datacentric_treshold = json.loads(args.datacentric_treshold)
    
    # Imprimir para verificação (opcional)
    print("Noise Level:", noise_level)
    print("Datacentric Treshold:", datacentric_treshold)

    from chosing_data_framework import Framework, decision
    
    if args.stage=='post':
        dataset_train = pd.read_csv(args.data_path_real + f'/{args.dataset_name}/{args.model}/dataset.csv',sep=';')
        print(dataset_train)
    else:
        dataset_train = pd.read_csv(args.data_path_real + f'/{args.dataset_name}/{args.dataset_name}_train.csv')
    full_dataiq, full_cleanlab = [], []
    dataiq_metrics, cleanlab_metrics = [], []

    for treshold in datacentric_treshold:
        print('TRESHOLD', treshold)
        for level in noise_level:
            print('LEVEL', level)

            framework = Framework(X_df=dataset_train,
                                noise_level=level,
                                datacentric_threshold=treshold)
            
            dataiq_metric, cleanlab_metric = framework.metrics()
            dataiq_metrics.append(dataiq_metric)
            cleanlab_metrics.append(cleanlab_metric)
            full_cleanlab.append([treshold, level, cleanlab_metric])
            full_dataiq.append([treshold, level, dataiq_metric])

    highest_dataiq = sorted(full_dataiq, key=lambda x: x[-1], reverse=True)[0]
    highest_cleanlab = sorted(full_cleanlab, key=lambda x: x[-1], reverse=True)[0]

    # Encontrar o maior valor final entre highest_dataiq e highest_cleanlab
    if highest_dataiq[-1] > highest_cleanlab[-1]:
        biggest = highest_dataiq
    else:
        biggest = highest_cleanlab

    dataiq_metrics = np.array(dataiq_metrics).flatten().tolist()
    cleanlab_metrics = np.array(cleanlab_metrics).flatten().tolist()

    # Imprimir as métricas e o maior valor encontrado para verificação
    print("DataIQ Metrics:", dataiq_metrics)
    print("Cleanlab Metrics:", cleanlab_metrics)
    print("Biggest:", biggest)

    # Chamar a função decision
    decision(path2save=args.path2save,
            dataiq_metrics=dataiq_metrics,
            cleanlab_metrics=cleanlab_metrics,
            bigger=biggest)

if __name__ == "__main__":
    main()
