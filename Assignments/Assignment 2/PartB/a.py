import csv
import numpy as np
import scipy as sc
import pandas as pd
import math
import sys
from numpy.linalg import inv

args = sys.argv

file1 = r'amazon_train.csv'
train = pd.read_csv(file1,header=None)
train = train.values

file2 = r'amazon_test_public.csv'
test = pd.read_csv(file2,header=None)	
test = test.values

print(len)