
import pandas as pd
import numpy as np 
from os import chdir

from adj_mat import weighted_directed_adj

df = pd.read_csv(r'coordinates.csv', index_col=0)

east= df['East']
north = df['North']
elevation = df['Elevation'] 

weighted_adj = weighted_directed_adj(east, north, elevation)
