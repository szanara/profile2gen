import numpy as np
import os
import pandas as pd

def read_data(path,file_name):
    complete_path = os.path.join(path, file_name)
    data = pd. read_csv(complete_path)
    return data 

