import pandas as pd
data = pd.read_csv('/home/ubuntu/gdh/MCPI2.0/celegans/protein.csv')
len_list =[]
for p in data['protein']:
    p_len = len(p)
    if p_len<1024:
        len_list.append(p_len)
    else:
        len_list.append(1024)
import pickle
with open('/home/ubuntu/gdh/MCPI2.0/celegans/protein_len.pkl','wb') as f:
    pickle.dump(len_list,f)