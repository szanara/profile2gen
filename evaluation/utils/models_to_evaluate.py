from autogluon.tabular import TabularPredictor
from functions import *

import os 
import pandas as pd 
class EvaluateData:
    def __init__(self, data_df_train, data_df_test, path2saveweights,path2saveperfs) -> None:
        self.data = data_df_train
        self.data_teste = data_df_test
        self.X_test =  (self.data_teste).drop(columns='target')
        self.y_test = (self.data_teste)['target']
        self.saving = path2saveweights
        self.savingperfs = path2saveperfs

    def trainining(self):
        predictor = TabularPredictor(label='target',problem_type='regression', path=self.saving).fit(self.data)
        return predictor
    
    def testing(self, predictor):
        predictor = TabularPredictor.load(self.saving)
        y_pred = predictor.predict(self.X_test)
        perf = predictor.evaluate_predictions(y_true=self.y_test, y_pred=y_pred, auxiliary_metrics=True)
        perf_all = predictor.leaderboard(self.data_teste)
        return perf, perf_all
    
    def evaluate_and_save(self):
        predictor = self.trainining()
        performance, performance_per_model = self.testing(predictor)
        os.makedirs(self.savingperfs,exist_ok=True)
        performance_lists = {key: [value] for key, value in performance.items()}
        # Criar DataFrame
        performance = pd.DataFrame(performance_lists)
        performance.to_csv(self.savingperfs+'/performance_media.csv')
        performance_per_model.to_csv(self.savingperfs+'/performance_per_model.csv')
        print(performance)
        print('SAVED')



class Augmenting:
    def __init__(self, df_train_real, df_train_synthetic) -> None:
        self.real = df_train_real
        self.synt = df_train_synthetic

    def random_choice(self, amount_required,seed):
        seed = seed
        amount_required = int(len(self.real) * amount_required)
        # Seleciona uma amostra aleatória dos dados sintéticos
        random_samples = self.synt.sample(n=amount_required, random_state=seed, replace=True)
        
        # Concatena os dados reais com os dados sintéticos selecionados
        dados_concatenados = pd.concat([self.real, random_samples])
        
        # Embaralha o DataFrame concatenado
        dados_concatenados_embaralhados = dados_concatenados.sample(frac=1, random_state=42)
        
        return dados_concatenados_embaralhados
    

    
    

        


        

