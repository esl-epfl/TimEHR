# %% [markdown]
# # Libraries


# from Utils.data import Physio3

# general
import numpy as np
import pandas as pd
import argparse

# import custom libraries
import sys
import os
import tqdm
import pickle
import yaml

# %%

# plotly
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


# %%
# folder paths

PATH_RAW="./raw/"
PATH_PROCESSED = "./processed/"
PATH_YAML = "../configs/data/"
# create folder if not exists
os.makedirs(PATH_RAW, exist_ok=True)
os.makedirs(PATH_PROCESSED, exist_ok=True)


# %%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.preprocessing import OneHotEncoder

# %%
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# %% [markdown]
# # Functions


class Physio3(Dataset):
    def __init__(self, all_masks,all_values,all_sta,static_processor=None, dynamic_processor=None, transform=None, ids=None, max_len=None):
        self.num_samples = all_masks.shape[0]
        self.mask = all_masks
        self.value = all_values
        self.sta = all_sta
        self.transform = transform

        self.static_processor = static_processor
        self.dynamic_processor = dynamic_processor
        self.ids = ids
        self.max_len = max_len

        self.n_ts = len(self.dynamic_processor['mean'])
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # image = self.data[idx]
        # label = self.labels[idx]

        # if self.transform:
        #     image = self.transform(image)

        return self.mask[idx], self.value[idx], self.sta[idx]
    

# %%
def create_df_demo(df_demo, demo_vars):
    from sklearn.preprocessing import OneHotEncoder
    demo_vars

    # print(demo_vars)
    # print(df.columns)

    # df_demo = df[['RecordID','Hospital']+demo_vars].dropna(subset='Age')
    # df_demo = df[['RecordID','Hospital']+demo_vars].dropna()
    # print(df_demo.columns)
    # term
    df_demo

    # replace -1 with NaN and the mean of each column
    df_demo.describe()
    df_demo[df_demo==-1]=np.nan
    df_demo.isnull().sum()
    df_demo.describe()
    for column in df_demo.columns:
        df_demo[column] = df_demo[column].fillna(df_demo[column].mean())
        # print(column)
    df_demo.isnull().sum()
    df_demo.describe()

    

    # standardize continuous variables
    demo_statistics={}
    for column in demo_vars:
        if column in ['Age', 'Height','Weight','HospAdmTime']:
            demo_statistics[column] = {'mean':df_demo[column].mean(),'std':df_demo[column].std()}

            df_demo[column] = (df_demo[column]-df_demo[column].mean())/df_demo[column].std()


    # discritze ICUType
    if 'ICUType' in demo_vars:

        ohe = OneHotEncoder()
        transformed = ohe.fit_transform(df_demo[['ICUType']])
        transformed
        mat_ICU_enc = pd.DataFrame.sparse.from_spmatrix(transformed).values.astype(int)
        new_cols=['ICUType'+str(i) for i in range(mat_ICU_enc.shape[1])]
        df_demo[new_cols]=mat_ICU_enc

        df_demo = df_demo.drop(columns='ICUType')
    
    if 'Gender' in demo_vars:
        df_demo['Gender'] = df_demo['Gender'].astype(int)  
         

    # df_demo.rename(columns={'RecordID':'id'},inplace=True)

    df_demo

    demo_vars_enc = list(df_demo.columns)
    demo_vars_enc.remove('RecordID')
    if 'Hospital' in demo_vars_enc:
        demo_vars_enc.remove('Hospital')
    
    print(demo_vars_enc)
    print(df_demo)
    df_demo['dict_demo'] = df_demo[demo_vars_enc].apply(lambda x:list(x),axis=1)

    df_demo.iloc[0]['dict_demo']


    dict_map_demos = {k:i for i,k in enumerate(demo_vars_enc)}
    dict_map_demos

    return df_demo, dict_map_demos, demo_statistics


# %%
def custom_process(df_demo, df_filt, train_ids,state_vars, gan_demovars):

    df_demo_train = df_demo[df_demo['RecordID'].isin(train_ids)].copy()
    df_filt_train = df_filt[df_filt['RecordID'].isin(train_ids)].copy()

    df_demo_test = df_demo[~df_demo['RecordID'].isin(train_ids)].copy()
    df_filt_test = df_filt[~df_filt['RecordID'].isin(train_ids)].copy()

    # if test is empty
    if df_demo_test.shape[0]==0:
        df_demo_test = df_demo_train.copy()
        df_filt_test = df_filt_train.copy()
    print('1',df_filt_train.isnull().sum().sum())

    # Normalize state variables

    # 1] normalized cols in state_vars from df_filt
    print('Step 1: Normalization ',df_filt_train.isnull().sum().sum())

    state_preprocess = {'mean':df_filt_train[state_vars].mean(),'std':df_filt_train[state_vars].std()}

    df_filt_train[state_vars] = (df_filt_train[state_vars]- state_preprocess['mean'])/state_preprocess['std']
    df_filt_test[state_vars] = (df_filt_test[state_vars]- state_preprocess['mean'])/state_preprocess['std']

    # 2] set outliers to nan
    print('Step 2: set outliers to nan ',df_filt_train.isnull().sum().sum())
    if DATASET not in ['energy','stock']:
        df_filt_train[state_vars] = df_filt_train[state_vars].apply(lambda x: x.mask(x.sub(x.mean()).div(x.std()).abs().gt(3)))
        df_filt_test[state_vars] = df_filt_test[state_vars].apply(lambda x: x.mask(x.sub(x.mean()).div(x.std()).abs().gt(3)))
        # print('step 2',df_filt_train.isnull().sum().sum())
        
    # 3] now do min-max normalization # between 0 and 1
    print('Step 3: min-max normalization ')
    state_preprocess['min'] = df_filt_train[state_vars].min()
    state_preprocess['max'] = df_filt_train[state_vars].max()

    df_filt_train[state_vars] = (df_filt_train[state_vars]-state_preprocess['min'])/(state_preprocess['max']-state_preprocess['min'])
    df_filt_test[state_vars] = (df_filt_test[state_vars]-state_preprocess['min'])/(state_preprocess['max']-state_preprocess['min'])

    # # # 4] scale to [-1,1]
    # print('Step 4: scale to [-1,1] ')
    # df_filt_train[state_vars] = df_filt_train[state_vars]*2-1
    # df_filt_test[state_vars] = df_filt_test[state_vars]*2-1


    # Normalize demo variables
    print('Step 5: Normalize demo variables ')
    df_demo_train, demo_dict,demo_preprocess = create_df_demo(df_demo_train, gan_demovars)

    demo_preprocess['demo_vars_enc'] = list(demo_dict.keys())
    df_demo_test, _,_ = create_df_demo(df_demo_test, gan_demovars)
    
    # print(list(demo_preprocess['demo_vars_enc']))
    return df_demo_train, df_demo_test, df_filt_train, df_filt_test, state_preprocess, demo_preprocess

# %%
def handle_cgan(df_demo, df_filt, state_vars,demo_vars_enc,granularity=1,target_dim=64):
   # var_old = [ 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
   #    'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
   #    'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na',
   #    'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets',
   #    'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine',
   #    'WBC']

   # print('set(state_vars)-set(var_old) => ',set(state_vars)-set(var_old))
   # print('set(var_old)-set(state_vars) => ',set(var_old)-set(state_vars))


   # df_demo, demo_dict = create_df_demo(df_demo, demo_vars)
   # print(demo_dict)
   # print(df_demo.columns)
   print(demo_vars_enc)
   sta = df_demo[demo_vars_enc].rename(columns={'In-hospital_death':'Label'})
   # sta['Label'] = sta['Label'].astype(float)
   # sta["Gender"] = sta["Gender"].astype(int)
   # sta["Label"] = sta["Label"].astype(int)

   # print(sta)
   all_sta = torch.from_numpy(sta.values)
   print(all_sta.shape)



   dyn=[]

   grouped = df_filt.groupby('RecordID')

   # Create a list of DataFrames
   dyn = [group[   ['Time']+state_vars].rename(columns={'Time':'time'}) for _, group in grouped]

   all_times = []
   all_dyn = []
   all_masks = []
   all_time_mark = []
   for SAMPLE in dyn:
      time = SAMPLE.time.values
      # times_padded = pd.DataFrame({'time':np.arange(0,   int(max(time)),granularity)})

      # print(time,len(time),max(time))
      # print(times_padded.values.flatten(),len(times_padded.values.flatten()))
      
      # create a np array from 0 to max(time) with granularity 0.5 ()including max(time)) use linspace
      time_padded = np.linspace(0, int(target_dim/granularity)-granularity, target_dim) # shape is (target_dim,)
      time_mark = time_padded<=max(time)
      # print(time_mark,time_mark.shape)
      # term
      df_time_padded = pd.DataFrame({'time':time_padded})
      temp = df_time_padded.merge(SAMPLE,how='outer',on='time')#.sort_values(by='time_padded').
      
      dyn_padded = temp[state_vars].fillna(0).values
      mask_padded = temp[state_vars].notnull().astype(int).values


      all_times.append(torch.from_numpy(time_padded))
      all_dyn.append(torch.from_numpy(dyn_padded))
      all_masks.append(torch.from_numpy(mask_padded))
      all_time_mark.append(torch.from_numpy(time_mark))
      # print(time_padded.shape,dyn_padded.shape,mask_padded.shape)
   
      
   all_masks = torch.stack(all_masks, dim=0)
   all_dyn = torch.stack(all_dyn, dim=0)
   all_times = torch.stack(all_times, dim=0)
   all_time_mark =torch.stack(all_time_mark)
   # PADDING
   # target_dim = TARGET_DIM
   padding_needed = target_dim - all_masks.shape[-1]

   all_masks_padded = torch.nn.functional.pad(all_masks, (0, padding_needed)).unsqueeze(1).float() # add channel dim
   all_dyn_padded = torch.nn.functional.pad(all_dyn, (0, padding_needed)).unsqueeze(1).float() # add channel dim
   all_times_padded = torch.nn.functional.pad(all_times, (0, padding_needed))
   all_masks_padded.shape
   all_dyn_padded.shape




   all_data = torch.stack([all_dyn_padded, all_masks_padded], dim=1)
   all_data.shape
   
   # # SCALE all_data to [-1,1]
   all_masks_padded = all_masks_padded*2-1
   all_dyn_padded = all_dyn_padded*2-1
   all_dyn_padded[all_masks_padded<0]=0
   # all_data = all_data*2-1
   # all_data[:,0,:,:][all_data[:,1,:,:]<0]=0


   
   return all_masks_padded.float(),all_dyn_padded.float(),all_sta.float(), all_time_mark.int()

# %% [markdown]
# # Simulated Dataset

# %%

def generate_poisson_binary_vector(length, lambda_value):
    # Generate random counts from a Poisson distribution
    poisson_counts = np.random.poisson(lambda_value, length)

    # Convert counts to a binary vector
    binary_vector = np.where(poisson_counts > 0, 1, 0)

    return binary_vector
def main():
    global DATASET
    for n_vars in opt.n_vars:
        for lambda_value in opt.lambdas:
            # %%
            # Set random seed for reproducibility
            np.random.seed(42)

            # Parameters
            # n_vars = 25
            # lambda_value = 2
            n_samples = 10000
            n_timestamps = 128
            period_min = 16
            period_max = 48
            TARGET_DIM = 128 if n_vars==128 else 64
            # TARGET_DIM = n_vars
            GRAN=1

            # Generate sinusoidal data with random phases
            data = np.zeros((n_samples, n_timestamps, n_vars))
            periods = np.random.randint(period_min, period_max + 1, n_vars)

            for var in range(n_vars):
                phase = np.random.uniform(0, 2*np.pi, 1)
                for sample in range(n_samples):
                    t = np.arange(n_timestamps)
                    amplitude = np.random.uniform(0.9, 1, 1)
                    noise = np.random.normal(0, 0.1, n_timestamps)
                    baseline = np.random.uniform(-0.5, 0.5, 1)
                    noise_period = np.random.randint(1, 5)
                    noise_phase = np.random.uniform(0, 2*np.pi/50, 1)
                    data[sample, :, var] = baseline + amplitude * np.sin(2 * np.pi * t / (periods[var]+noise_period) + phase) + noise


            # Simulate random missing values based on a Poisson process
            missing_mask = np.array([ generate_poisson_binary_vector(n_timestamps, lambda_value) for i in range(n_samples*n_vars)]).reshape(n_samples,n_vars,n_timestamps).transpose(0,2,1)

            mr = (missing_mask==0).sum() / missing_mask.size*100
            print(f'missing rate: {mr:.2f}%')

            # apply missing mask
            data_with_missing = np.where(missing_mask == 0, np.nan, data)





            # Create a plot using Plotly
            # You may need to install plotly via pip if you haven't already: pip install plotly
            sample_idx = 0  # Choose a sample to plot
            fig = go.Figure()

            for var in range(5):
                _ = fig.add_trace(go.Scatter(x=np.arange(n_timestamps), y=data_with_missing[sample_idx, :, var],
                                        mode='lines', name=f'Variable {var+1}'))

            _ = fig.update_layout(title='Simulated Multivariate Time Series with Missing Data',
                            xaxis_title='Time Steps', yaxis_title='Value')
            # fig.show()

            data.shape
            data_with_missing.shape


            # %%
            # convert to dataframe

            col_names = [f'var_{i}' for i in range(n_vars)]

            df = pd.DataFrame(data_with_missing.reshape(-1, n_vars), columns=col_names)


            df.to_csv(PATH_RAW+f'sim-l{lambda_value}-d{n_vars}.csv',index=False)

            # %%
            df.shape
            df.head()
            n_vars

            data.reshape(-1, n_vars).shape

            # %%
            corr_mat = df.corr().values
            corr_mat[np.abs(corr_mat)<0.2] = 0
            corr_mat[np.triu_indices_from(corr_mat)] = 0

            corr_mat.shape, len(col_names)
            fig = px.imshow(corr_mat,x=col_names,y=col_names, color_continuous_scale='RdBu', zmin=-1, zmax=1, color_continuous_midpoint=0)

            # fig.show()





            # %% [markdown]
            # # Original Dataset


            DATASET = f'sim-l{lambda_value}-d{n_vars}'
            SUFFLE_VARS = False

            print(DATASET)




            df = pd.read_csv(PATH_RAW+f'{DATASET}.csv')
            seq_len = 128 if '128' in DATASET else 64
            # if '16' in DATASET:
            #     seq_len = 16
            # elif '32' in DATASET:
            #     seq_len = 32
            # elif '64' in DATASET:
            #     seq_len = 64
            # elif '128' in DATASET:
            #     seq_len = 128
            

            # MODE 1: disjoint

            a = len(df)//seq_len*seq_len
            df = df.iloc[:a]
            # simplify above

            var_names = df.columns.tolist()

            df['Time'] =  np.tile(np.arange(seq_len),len(df)//seq_len)
            df['RecordID'] = np.repeat(np.arange(len(df)//seq_len),seq_len)
            
            
            
            # # MODE 2: overlapping
            # mat = df.values
            # # Preprocess the dataset
            # temp_data = []
            # id_data = []
            # time_data = []
            # # Cut data by sequence length
            # for i in range(0, len(mat) - seq_len):
            #     _x = mat[i:i + seq_len]
            #     temp_data.append(_x)
            #     id_data.append(np.array(np.ones(seq_len)*i))
            #     time_data.append(np.arange(seq_len))
            # len(temp_data),temp_data[0].shape

            # mat2 = np.concatenate(temp_data)
            # mat2.shape

            # id_data = np.concatenate(id_data)
            # id_data.shape
            
            # time_data = np.concatenate(time_data)
            # time_data.shape
            #     #form a dataframe
            # df = pd.DataFrame(mat2, columns=df.columns)
            # df['RecordID'] = id_data
            # df['Time'] = time_data
            
            



            # continue
            
            df['Hospital'] = 0
            # set age randomly
            df['Age'] = np.random.randint(20,80,len(df))     
            df['Label'] = 0
            all_cols = df.columns

            cols_id = ['RecordID','Hospital','Time']
            cols_outcome = ['Label']
            cols_demo = ['Age']
            cols_vital = var_names
            cols_lab = []
            cols_ignore = []



            # %%
            remaining_cols = list(set(all_cols) - set(cols_id) - set(cols_outcome) - set(cols_demo) - set(cols_vital) - set(cols_lab) - set(cols_ignore))

            print("remaining_cols: ", remaining_cols)
            print("number of patients: ", df.RecordID.nunique())

            # %%
            state_vars = cols_vital+cols_lab

            if SUFFLE_VARS:
                np.random.seed(42)
                np.random.shuffle(state_vars)

            state_vars[:5]

            event_vars = cols_lab
            demo_vars = cols_demo

            dict_map_states = {label:i for i,label in enumerate(state_vars)}


            dict_map_events = {label:i for i,label in enumerate(event_vars)}

            dict_map_demos = {k:i for i,k in enumerate(cols_demo)}
            dict_map_demos


            print("number of state variables: ", len(state_vars))

            # %%
            # split df to df_ts and df_static

            df['RecordID'] = df['RecordID'].astype(int)
            df['Label'] = df['Label'].astype(int)

            df_ts = df[['RecordID','Time']+cols_vital+cols_lab].copy()
            df_static = df[['RecordID']+cols_demo+cols_outcome].drop_duplicates(subset='RecordID').copy()



            

            # %%
            # choose time granularity
            time_granularity = 1  # 1 hour
            df_ts.iloc[:30].Time.values
            df_ts['Time'] = df_ts['Time'].apply(lambda x: round(x/time_granularity,0)*time_granularity)
            df_ts.iloc[:30].Time.values
        

            df_ts = df_ts.dropna(subset=cols_vital+cols_lab, how='all')






            # %%
            # is aggregation needed?

            a1 = df_ts.groupby('RecordID').size().values
            a2 = df_ts.groupby('RecordID')['Time'].nunique().values

            if (a1-a2).sum()>0:
                df_ts.shape
                # forwardfill for each group

                df_ts[cols_lab+cols_vital] = df_ts[['RecordID','Time']+cols_lab+cols_vital].groupby(['RecordID','Time']).fillna(method='ffill')
                df_ts.shape

                # keep last for each group
                df_ts = df_ts[['RecordID','Time']+cols_lab+cols_vital].groupby(['RecordID','Time']).last().reset_index()
                df_ts.shape
            else:
                print('no aggregation needed')


            # forwarfill for each group
            df_static[cols_demo+cols_outcome] = df_static[['RecordID']+cols_demo+cols_outcome].groupby(['RecordID']).fillna(method='ffill')

            # keep last for each group
            df_static = df_static[['RecordID']+cols_demo+cols_outcome].groupby(['RecordID']).last().reset_index()

            df_static.shape

            # %%
            df_ts.RecordID.nunique(), df_static.RecordID.nunique()

            # %% [markdown]
            # ## K-fold

            # %%
            N=5

            df_static.shape
            # seed for numpy and dataframe sampling
            np.random.seed(42)


            # shuffle ids
            list_ids = df_static['RecordID'].values.copy()
            np.random.shuffle(list_ids)
            list_ids[:5]

            df_static = df_static[df_static['RecordID'].isin(list_ids)]
            df_static.head()



            split_list = np.linspace(0,len(df_static),5+1).astype(int)
            split_list




            
            i_split=0
            path2save = PATH_PROCESSED+f'{DATASET}/split{i_split}/'
            os.makedirs(path2save , exist_ok=True)
            
            test_ids = list_ids[split_list[i_split]:split_list[i_split+1]]
            i_split, sum(test_ids)
            train_ids = list(set(list_ids)-set(test_ids))

            # if DATASET=='energy':
            #     train_ids = list_ids
                
            # save train ids
            with open(path2save+'train_ids.pkl', 'wb') as f:
                pickle.dump(train_ids, f)

        

            # save df_static and df_ts
            df_static.to_csv(path2save+'df_static.csv',index=False)
            df_ts.to_csv(path2save+'df_ts.csv',index=False)
            
            print('saved to', path2save)
                






            gan_demovars = cols_demo+cols_outcome




            
            
            path2save = PATH_PROCESSED+f'{DATASET}/split{i_split}/'
            
            print("path2save:", path2save)

            
            df_static = pd.read_csv(path2save+'df_static.csv', index_col=None)    
            df_ts = pd.read_csv(path2save+'df_ts.csv', index_col=None)
            df_static.shape, df_ts.shape

            
            # # only keep the ids in split_ids
            # df_static = df_static[df_static['RecordID'].isin(split_ids)].copy()
            # df_ts = df_ts[df_filt['RecordID'].isin(split_ids)].copy()
            
            df_ts = df_ts.merge(df_static[['RecordID','Label']],on=['RecordID'],how='inner') # add death column to df_ts
            df_static = df_static.merge(df_ts[['RecordID']].drop_duplicates(),on=['RecordID'],how='inner') # add hospital 
            
            # sort both
            df_static = df_static.sort_values(by=['RecordID'])
            df_ts = df_ts.sort_values(by=['RecordID','Time'])

            
            df_static.shape, df_ts.shape
            
            # df_ts.isnull().sum().sum()
            # continue
            
            # load split_ids
            with open(path2save+'train_ids.pkl', 'rb') as f:
                train_ids = pickle.load(f)
                len(train_ids)

            # load dev_ids
            # with open(path2save+'dev_ids.pkl', 'rb') as f:
            #     dev_ids = pickle.load(f)
            dev_ids=[-5555]
            len(dev_ids)
            
            # exclude dev_ids from train_ids
            df_static = df_static[~df_static['RecordID'].isin(dev_ids)].copy()
            df_ts = df_ts[~df_ts['RecordID'].isin(dev_ids)].copy()
            
            df_static.shape, df_ts.shape

            
            
            df_static_train, df_static_test, df_ts_train, df_ts_test,   state_preprocess, demo_preprocess  = custom_process(df_static, df_ts, train_ids, state_vars, gan_demovars)
            
            
            

            

            df_static_train.shape, df_static_test.shape
            df_ts_train.shape, df_ts_test.shape

            
            
            
            demo_vars_enc = demo_preprocess['demo_vars_enc']

            # Train dataset
            all_masks,all_values,all_sta,all_time_mark = handle_cgan(df_static_train, df_ts_train,state_vars,demo_vars_enc,granularity=GRAN,
            target_dim=TARGET_DIM)

            
            
            ph = Physio3(all_masks,all_values,all_sta,static_processor=demo_preprocess, dynamic_processor=state_preprocess, ids=df_static_train['RecordID'].values, max_len=all_time_mark)  

            with open(path2save+f"/train.pkl", 'wb') as file:
                pickle.dump(ph, file)       

            # Val dataset
            all_masks,all_values,all_sta,all_time_mark = handle_cgan(df_static_test, df_ts_test,state_vars,demo_vars_enc,granularity=GRAN,target_dim=TARGET_DIM)
            temp2 = all_sta.clone()

            
            
            ph = Physio3(all_masks,all_values,all_sta,static_processor=demo_preprocess,
            dynamic_processor=state_preprocess, ids=df_static_test['RecordID'].values, max_len=all_time_mark)    

            with open(path2save+f"/eval.pkl", 'wb') as file:
                pickle.dump(ph, file) 

            # cgan3 now pixels are from -1 to 1

            # create the yaml file

            yaml_dict = {
                'name': DATASET,
                'path_raw': f'./data/raw',
                'path_processed': f'./data/processed/{DATASET}',
                'path_train': f'./data/processed/{DATASET}/split{{SPLIT}}/train.pkl',
                'path_eval': f'./data/processed/{DATASET}/split{{SPLIT}}/eval.pkl',
                'img_size': TARGET_DIM,
                'd_static': 2
            }

            with open(PATH_YAML+f'{DATASET}.yaml', 'w') as file:
                yaml.dump(yaml_dict, file)

    

if __name__ == "__main__":

    DATASET = 'sim'
    parser = argparse.ArgumentParser(description='Data Preprocessing')

    # # parser.add_argument('--dataset', type=str, default='p12', help='dataset', dest="DATASET" )
    # parser.add_argument('--d', type=int, default=32, help='number of variables', dest="n_vars")
    # parser.add_argument('--lambda', type=float, default=0.5, help='number of variables', dest="lambda_value")
    parser.add_argument('--n-vars', nargs='+', help='A list of integers.', type=int,default=[32], dest='n_vars')
    parser.add_argument('--lambdas', nargs='+', help='A list of floats.', type=float,default=[0.5], dest='lambdas')


    opt = parser.parse_args()

    print(opt.n_vars)
    print(opt.lambdas)
    
    # lambda_value = opt.lambda_value
    # n_vars = opt.n_vars

    # DATASET = f'sim-l{lambda_value}-d{n_vars}'
    # SUFFLE_VARS = False
    pass
    main()

