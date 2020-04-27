import pandas as pd
import subprocess
import tensorflow as tf
import numpy as np
import time
import os


def get_name_size(data):
    input_list = []
    output_list = []
    result = re.search(r"=== Automatically deduced input nodes ===.*?===", data, re.S)

    if result:
        input_data = result.group(0)

        names = re.findall(r"name.*?\,|name.*?\]", input_data, re.S)

        if len(names) > 0:
            for name in names:

                info = re.findall(r"name.*|size.*", name, re.M)
                if len(info) > 0:
                    name_list = []

                    for val in info:
                        pos = val.find(":")
                        if pos >= 0:
                            name_list.append(val[pos + 1:].replace('"', '').strip(" "))
                    input_list.append(name_list)

    result = re.search(r"=== Automatically deduced output nodes ===.*?===", data, re.S)

    if result:
        output_data = result.group(0)

        names = re.findall(r"name.*?\,|name.*?\]", output_data, re.S)

        if len(names) > 0:
            for name in names:

                info = re.findall(r"name.*", name, re.M)
                if len(info) > 0:

                    for val in info:
                        pos = val.find(":")
                        if pos >= 0:
                            output_list.append(val[pos + 1:].replace('"', '').strip(" "))

    return input_list, output_list


# pd convert onnx and return log str
def pb2onnx(pb_name, onnx_name, input_nodes_str, output_nodes_name):
    input_nodes_list = eval(input_nodes_str)  # str=>list
    input_params = None  # concat uffInput params
    for node in input_nodes_list:
        input_params = input_params + node[0] + ","
        node_size = node[1]
        if len(node_size) == 4:
            input_params = input_params + '{},{},{},'.format(node_size[3], node_size[1], node_size[2])
        elif len(node_size) == 2:
            input_params = input_params + '1,{},{},'.format(node_size[1], node_size[1])
        input_params = input_params[:-1]  # remove redundent comma
    return subprocess.run("python3 -m  tf2onnx.convert --input {}  --output {} --inputs {}  --outputs {}".format(pb_name, onnx_name, input_params, output_nodes_name), shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
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
    pb_name = table.iloc[seq].task
    onnx_name = os.path.basename(pb_name).split('.')[0] + ".onnx"
    log_pb2onnx = pb2onnx(pb_name, onnx_name,table.iloc[seq].input_nodes, table.iloc[seq].output_nodes)
    log_onnxtest = onnx_test(onnx_name, get_name_size(log_pb2onnx)[0])
    # write log
    log_file = os.path.join(logs_dir, os.path.basename(pb_name).split('.')[0] + ".txt")
    if os.path.isfile(log_file):
        os.remove(log_file)
    with open(log_file, 'w') as f:
        f.write(log_pb2onnx)
        f.write('\n')
        f.write(log_onnxtest)
table.to_csv('table.csv')