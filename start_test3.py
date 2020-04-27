import pandas as pd
import subprocess
import tensorflow as tf
import numpy as np
import time
import os

# pd convert onnx and return log str
def pb2onnx(pb_name, onnx_name, input_nodes_name, output_nodes_name):
    return subprocess.run("python3 -m  tf2onnx.convert --input {}  --output {} --inputs {}  --outputs {}".format(pb_name, onnx_name, input_nodes_name, output_nodes_name), shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
# onnx test,return log str
def onnx_test(onnx_name,input_nodes_shape):
    return subprocess.run("./trtexec --onnx= {} --shapes={}".format(onnx_name,input_nodes_shape), shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')


table = pd.read_csv('table.csv')
logs_dir = './test/'
# create log directory
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
def start_test(table ,seq):
    pb_name = table.iloc[seq].task
    onnx_name = os.path.basename(pb_name).split('.')[0] + ".onnx"
    log_pb2onnx = pb2onnx(pb_name, onnx_name,table.iloc[seq].input_nodes, table.iloc[seq].output_nodes)
    input_nodes_shape = "???????"
    log_onnxtest = onnx_test(onnx_name,input_nodes_shape)
    # write log
    log_file = os.path.join(logs_dir, os.path.basename(pb_name).split('.')[0] + ".txt")
    if os.path.isfile(log_file):
        os.remove(log_file)
    with open(log_file, 'w') as f:
        f.write(log_pb2onnx)
        f.write('\n')
        f.write(log_onnxtest)
table.to_csv('table.csv')