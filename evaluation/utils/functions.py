
import pandas as pd
import numpy as np
import os

class PrepareDataToGen:
    def __init__(self,data_csv) -> None:
        self.data = data_csv

    def columns_types(self)-> dict:
        types = {'string': [], 'int': [], 'float': []}

        for coluna in (self.data).columns:
            tipo = self.data[coluna].dtype

            if tipo == 'object':
                types['string'].append(coluna)
            elif tipo == 'int64':  # 'int64' é o tipo para inteiros no pandas
                types['int'].append(coluna)
            elif tipo == 'float64':  # 'float64' é o tipo para floats no pandas
                types['float'].append(coluna)
            else:
                try:
                    # Tenta converter para int
                    self.data[coluna].astype(int)
                    types['int'].append(coluna)
                except ValueError:
                    try:
                        # Tenta converter para float
                        self.data[coluna].astype(float)
                        types['float'].append(coluna)
                    except ValueError:
                        # Se não pode ser convertido para int ou float, considera como string
                        types['string'].append(coluna)

        return types
    
    def withoutnan(self):
        nans_linhas = (self.data).isna().any(axis=1).any()
        nans_colunas = (self.data).isna().any(axis=0).any()

        # Se houver valores NaN, substitui por zero
        if nans_linhas or nans_colunas:
            self.data = (self.data).fillna(0)
        return self.data


    def filtering_dgan(self):
        self.data = self.withoutnan()
        types = self.columns_types()
        self.data = (self.data).drop(columns=types['string'] )

        df_int = pd.DataFrame(index=(self.data).index)
        df_float = pd.DataFrame(index=(self.data).index)

        for coluna in (self.data).columns:
            try:
                # Tenta converter para int
                df_int[coluna] = self.data[coluna].astype(int)
            except ValueError:
                df_float[coluna] = self.data[coluna].astype(float)
        return [df_int, df_float]

    def filtering_transformer(self):
         # Remove rows with NaN values
        self.data = self.withoutnan()  # Assuming this already removes rows with NaNs
    
        # Remove rows where any value is zero
        #self.data = self.data[(self.data != 0).all(axis=1)]
    
        # Identify column types and drop string columns
        types = self.columns_types()
        self.data = self.data.drop(columns=types['string'])
        return self.data





        