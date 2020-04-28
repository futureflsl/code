import pandas as pd
import subprocess
import tensorflow as tf
import numpy as np
import time
import os

# onnx test,return log str
def onnx_test(onnx_name,input_nodes_str):
    input_nodes_list = eval(input_nodes_str)  # str=>list
    input_params = None  # concat uffInput params
    for node in input_nodes_list:
        input_params = input_params +"'{}':".format(node[0])
        node_size = node[1]
        if len(node_size) == 4:
            input_params = input_params + '{}x{}x{}x{},'.format(node_size[0], node_size[1], node_size[2],node_size[3])
        elif len(node_size) == 2:
            input_params = input_params + '1x{}x{}x1,'.format(node_size[1], node_size[1])
        input_params = input_params[:-1]  # remove redundent comma
    return subprocess.run("./trtexec --onnx= {} --shapes={}".format(onnx_name,input_params), shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')


table = pd.read_csv('table.csv')
logs_dir = './test/'
# create log directory
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
def start_test(table ,seq):
    pb_name = table.iloc[seq].tasks
    input_nodes_str = table.iloc[seq].input_nodes
    onnx_name = os.path.basename(pb_name).split('.')[0] + ".onnx"
    log_onnxtest = onnx_test(onnx_name, input_nodes_str)
    # write log
    log_file = os.path.join(logs_dir, os.path.basename(pb_name).split('.')[0] + ".txt")
    if os.path.isfile(log_file):
        os.remove(log_file)
    with open(log_file, 'w') as f:
        f.write(log_onnxtest)
table.to_csv('table.csv')