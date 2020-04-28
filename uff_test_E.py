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


def start_test(table, seq, batches):
    uff_name = table.iloc[seq].tasks
    input_nodes_str = table.iloc[seq].input_nodes
    output_nodes_name = table.iloc[seq].output_node
    for batch in batchs:
        input_nodes_list = eval(input_nodes_str)  # str=>list
        input_params = ""  # concat uffInput params
        for node in input_nodes_list:
            input_params += " --input="+node[0] + ","
            node_size = node[1]
            node_size = node_size[1:]
            for size in node_size:
                input_params += size + ","
            input_params = input_params[:-1]  # remove redundent comma
            command = './trtexec --uff={} --output={}{} --batch={}'.format(uff_name, output_nodes_name, input_params, batch)
            log = subprocess.run(command, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
            # write log
            log_file = os.path.join(logs_dir, os.path.basename(uff_name).split('.')[0] + "_uff.txt")
            if os.path.isfile(log_file):
                os.remove(log_file)
            with open(log_file, 'w') as f:
                f.write(log)
table.to_csv('table.csv')
