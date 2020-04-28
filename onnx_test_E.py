import pandas as pd
import subprocess
import tensorflow as tf
import numpy as np
import time
import os


table = pd.read_csv('table.csv')
logs_dir = './test/'
# create log directory
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
def start_test(table ,seq):
    pb_name = table.iloc[seq].tasks
    input_nodes_str = table.iloc[seq].input_nodes
    input_nodes_names = eval(input_nodes_str)
    shapes=' '.join(["--shapes=\'"+i[0]+":0\':"+ 'x'.join(i[1]).replace('-','') for i in input_nodes_names])
    onnx_name = os.path.basename(pb_name).split('.')[0] + ".onnx"
    command = './trtexec --onnx={} {} --fp16'.format(onnx_name, shapes)
    log_onnxtest = subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
    # write log
    log_file = os.path.join(logs_dir, os.path.basename(pb_name).split('.')[0] + ".txt")
    if os.path.isfile(log_file):
        os.remove(log_file)
    with open(log_file, 'w') as f:
        f.write(log_onnxtest)
start_test(table ,1)
table.to_csv('table.csv')