import numpy as np
import pandas as pd

file_path = '/home/varun/catkin_ws/src/hero_plus/hero_bringup/config/30secondbehaviour.csv'
data = pd.read_csv(file_path)

y=data.iloc[:,-1:]
print(y)