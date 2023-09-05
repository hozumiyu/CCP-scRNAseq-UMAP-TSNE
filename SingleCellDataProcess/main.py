# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:42:44 2023

@author: yutah
"""

import os, sys


data = sys.argv[1]
try:
    outpath = sys.argv[2]
    print('Outpath found. Data will be saved at', outpath)
except:
    outpath = './data/'
    print('No out path found. Using default:', outpath)
    
try:
    data_process_path = sys.argv[3] 
    print('Outpath found. Data will be saved at', data_process_path)
except:
    data_process_path = './'
    print('No out path found. Using default:', data_process_path)

if data[:8] in 'GSE84133':
    data = 'GSE84133'

os.system('python %s/code//%s_process.py %s %s'%(data_process_path, data, outpath, data_process_path))