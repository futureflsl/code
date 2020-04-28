
import pandas as pd

with open('all_nodes.txt','r')as f:
    all_nodes=eval(f.read())
with open('assignment.txt','r')as f:
    tasks=f.read().split('\n')
tasks=[task for task in tasks if task!='']



output_node=[node[1][0] for node in all_nodes]

def wash(nas):
    nas=nas[0]
    result=[(na[0],na[1:])for na in nas]
    return result
input_nodes_shape=[wash(node) for node in all_nodes]

n_tasks_pb=[task[:-3]+'_FP16.pb' for task in tasks]
n_tasks_uff=[task[:-3]+'_FP16.uff' for task in tasks]
n_tasks_onnx=[task[:-3]+'_FP16.onnx' for task in tasks]
tasks.extend(n_tasks_pb)
tasks.extend(n_tasks_uff)
tasks.extend(n_tasks_onnx)
table=pd.DataFrame()
table['tasks']=tasks
table['output_node']=output_node*4
table['input_nodes']=input_nodes_shape*4
table['step_time']=0
table.to_csv('table.csv')
