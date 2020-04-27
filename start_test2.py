import pandas as pd
import subprocess
import tensorflow as tf
import numpy as np
import time
import os

table = pd.read_csv('table.csv')
tasks = table.tasks.tolist()
tasks = [task for task in tasks if os.path.splitext(task)[1] == '.uff']
batchs = [1, 2, 5]
# ./trtexec --uff=e_trainingFalse.opt.uff --output=latents_out --uffInput=images_in,512,512,3     --uffNHWC（可选） --batch =8
logs_dir = './test/'
# create log directory
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# excute trtexec command by subprocess,return log information
def exec_uff(uff_name, input_nodes_str,batch):
    input_nodes_list = eval(input_nodes_str) # str=>list
    input_params=None # concat uffInput params
    for node in input_nodes_list:
        input_params=input_params+node[0]+","
        node_size = node[1]
        if len(node_size)==4:
            input_params=input_params+'{},{},{},'.format(node_size[3],node_size[1],node_size[2])
        elif len(node_size)==2:
            input_params=input_params+'1,{},{},'.format(node_size[1],node_size[1])
        input_params = input_params[:-1] # remove redundent comma
    return subprocess.run('./trtexec --uff={} --output=latents_out --uffInput={} --uffNHWC --batch ={}'.format(uff_name,input_params,batch), shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')

def start_test(table, seq, batches):
    uff_name = table.iloc[seq].task
    input_nodes_str = table.iloc[seq].input_nodes
    for batch in batchs:
        log = exec_uff(uff_name, input_nodes_str, batch)
        # write log
        log_file = os.path.join(logs_dir, os.path.basename(uff_name).split('.')[0] + "_uff.txt")
        if os.path.isfile(log_file):
            os.remove(log_file)
        with open(log_file, 'w') as f:
            f.write(log)
table.to_csv('table.csv')
