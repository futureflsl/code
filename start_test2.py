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
def start_test():
    for task in tasks:
        for batch in batchs:
            log = subprocess.run('./trtexec --uff={} --output=latents_out --uffInput=images_in,512,512,3 --uffNHWC --batch ={}'.format(task,batch), shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
            # write log
            log_file = os.path.join(logs_dir, os.path.basename(task).split('.')[0] + ".txt")
            if os.path.isfile(log_file):
                os.remove(log_file)
            with open(log_file, 'w') as f:
                f.write(log)
table.to_csv('table.csv')
