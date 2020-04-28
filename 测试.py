inputs = [['input_ids', '-1', '320'],
 ['input_mask', '-1', '320'],
 ['segment_ids', '-1', '320']]

input_params = '--inputs ' + ',inputs='.join([i[0]+":0" for i in inputs]).replace('-', "")
input_params = input_params.replace("inputs=", " --inputs ")
print(input_params)