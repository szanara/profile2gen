#THIS CODE OR THE BIGGER PART OF IT CAME FROM THE REPOSITORY  https://github.com/HLasse/data-centric-synthetic-data
'''CODE EMPLYED TO PICK THE BEST FRAMEWORK TO PROFILE DATA'''

 

import os
import numpy as np
import sys 
sys.path.append('YOUR PATH TO UTILS/utils')
from noise import *
from sculping_data import *
from dataiqreg import *
from sklearn.linear_model import LinearRegression




class Framework:
    def __init__(self, X_df,noise_level, datacentric_threshold) -> None:
        self.X= X_df.drop(columns='target')
        self.y = X_df['target']
        self.level = noise_level
        self.thresold = datacentric_threshold
    

    def fake_labels(self):

        labels_modified,noise_indices = introduce_label_noise(labels=self.y,
                                                      noise_level=self.level)
        return labels_modified, noise_indices
    
    def getting_had_indices(self, labels_modified):

        
        X_easy_cleanlab,X_ambiguous_cleanlab,X_hard_cleanlab,y_easy_cleanlab, y_ambigious_cleanlab,y_hard_cleanlab,indices_cleanlab, percentile_threshold_cleanlab,data_centric_threshold_cleanlab =sculpt_with_cleanlab(
        X = self.X,
        y = labels_modified,
        data_centric_threshold= self.thresold
        )  
        
        X_easy_dataiq,X_ambiguous_dataiq,X_hard_dataiq,y_easy_dataiq, y_ambigious_dataiq,y_hard_dataiq,indices_dataiq, percentile_threshold_dataiq,data_centric_threshold_dataiq  = sculpt_with_dataiq(
        X = self.X,
        y = labels_modified,
        data_centric_threshold =self.thresold,
        method ="dataiq",
        device='cuda',
        epochs=10,
        percentile_threshold = None
        )

        indices_instance_cleanlab = indices_cleanlab[1]
        # Acessando a parte 'hard':
        hard_indices_cleanlab = indices_instance_cleanlab.hard

        indices_instance_dataiq= indices_dataiq[1]
        # Acessando a parte 'hard':
        hard_indices_dataiq = indices_instance_dataiq.hard

        return  hard_indices_cleanlab, hard_indices_dataiq

    def metrics(self,):
        labels_modified, noise_indices = self.fake_labels()
        hard_indices_cleanlab, hard_indices_dataiq = self.getting_had_indices(labels_modified)
       
        # Contar quantos elementos da lista menor estão na lista maior
        contador_cleanlab = len(set(hard_indices_cleanlab).intersection(noise_indices))
        contador_dataiq = len(set(hard_indices_dataiq).intersection(noise_indices))

        dataiq_metric = calculate_metrics(n_correct_hard = contador_dataiq, 
                  n_flipped =len(noise_indices) ,
                  n_hard =len(set(hard_indices_dataiq)))
        cleanlab_metric = calculate_metrics(n_correct_hard = contador_cleanlab, 
                  n_flipped =len(noise_indices) ,
                  n_hard =len(set(hard_indices_cleanlab)))
        
        return dataiq_metric, cleanlab_metric
    

def decision(path2save: str, dataiq_metrics: list, cleanlab_metrics: list, bigger: list):
    # Calcule a média das métricas para cada framework
    avg_pfmce_dataiq = sum(dataiq_metrics) / len(dataiq_metrics)
    avg_pfmce_cleanlab = sum(cleanlab_metrics) / len(cleanlab_metrics)
    
    threshold_touse = bigger[0]
    
    # Armazene a média e o nome de cada framework em uma lista
    mean_dataiq = [avg_pfmce_dataiq, 'dataiq', threshold_touse]
    mean_cleanlab = [avg_pfmce_cleanlab, 'cleanlab', threshold_touse]
    
    # Verifique se há diferença entre as médias
    if avg_pfmce_dataiq != avg_pfmce_cleanlab:
        # Escolha o framework com a melhor média
        best = max(mean_cleanlab, mean_dataiq, key=lambda x: x[0])
        print(f"O framework {best[1]} é o melhor com uma média de {best[0]} e threshold {best[2]}")
    else:
        print("Ambos os frameworks têm a mesma média")
        best = mean_cleanlab  # Pode escolher qualquer um já que as médias são iguais
    
    # Crie o caminho completo para salvar o arquivo
    
    pth_complete = os.path.join('PATH2SAVE/models/frameworks', path2save)
    os.makedirs(pth_complete, exist_ok=True)
    
    # Salve o melhor framework no arquivo .npy
    np.save(os.path.join(pth_complete, 'best_framework.npy'), best)
    
    return best

 