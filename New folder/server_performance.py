#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# from matplotlib.pylab import rcParams
# rcParams['figure.figsize']=20,10

# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler(feature_range=(0,1))
 
# from keras.models import Sequential
# from keras.layers import LSTM,Dropout,Dense


# In[3]:


df = pd.read_csv('DataCollector01.csv');
# df.head();
# df.info();


# In[4]:


# df.info()


# In[5]:


df['Time']
df = df.drop(0)


# In[6]:


df = df.astype({"Time": str})


# In[7]:


df['Time'] = df['Time'].str.replace(r'11/14/2022', '')


# In[8]:


df['Time']


# In[9]:


dataset = df.copy()


# In[10]:


# df = pd.DataFrame(dataset, columns = ['Time', 'Memory', 'Disk','CPU'],
#                   index = ['a', 'b', 'c', 'd'])
# df.drop([df.index[0]])


# In[11]:


dataset = dataset.astype({"Memory": float,"Disk":float,"CPU":float})
dataset.info()


# In[12]:


dataset['performance'] = (dataset['Memory']+dataset['Disk']+dataset['CPU'])/3


# In[13]:


dataset.describe()


# In[14]:


dataset1 = dataset.iloc[:10]
dataset1.index=dataset1['Time']
plt.figure(figsize=(16,8))
# dataset1
plt.plot(dataset1["performance"],label='Close Price history')


# In[15]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
print(f"rows in train set:{len(train_set)}\n rows in test set : {len(test_set)}")


# In[16]:


from pandas.plotting import scatter_matrix
# dataset_label
attributes= ["Time","Memory", "Disk","CPU"]
scatter_matrix(dataset[attributes], figsize = (12,8))


# In[17]:


dataset1 = dataset.iloc[:10]
dataset1.index=dataset1['Time']
plt.figure(figsize=(16,8))
dataset1
# plt.plot(dataset1["performance"],label='Close Price history')


# In[18]:


dataset = train_set.drop("performance", axis=1)
dataset_label = train_set['performance'].copy()
dataset_label


# In[19]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(fill_value=None)
dataset_w_t = dataset.drop("Time",axis = 1)
dataset_time = dataset['Time'].copy()
# SimpleImputer(missing_values=np.nan
imputer.fit(dataset_w_t)
dataset_time


# In[20]:


X = imputer.transform(dataset_w_t)


# In[21]:


dataset_tr = pd.DataFrame(X, columns=dataset_w_t.columns)


# ## Creating Pipeline

# In[22]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])


# In[23]:


dataset_arr = my_pipeline.fit_transform(dataset_tr)


# In[24]:


dataset_arr.shape


# ## Selecting a desired model

# In[25]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor()
model = DecisionTreeRegressor()
# model = LinearRegression()
model.fit(dataset_arr,dataset_label)


# In[26]:


some_data = dataset_tr.iloc[:5]
some_labels = dataset_label.iloc[:5]


# In[27]:


prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[28]:


some_labels


# ## Evaluating the model

# In[29]:


from sklearn.metrics import mean_squared_error
performance_predictions = model.predict(dataset_w_t)
lin_mse = mean_squared_error(dataset_label, performance_predictions)
lin_rms = np.sqrt(lin_mse)


# In[30]:


lin_rms


# ## testing the model on test data

# In[33]:


# x_test = test_set.drop('performance', axis=1)
x_test= test_set.drop({'Time','performance'}, axis = 1)
y_test = test_set['performance'].copy()
# x_test = 
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# unit_time_per = final_predictions/5
print(final_predictions[:1], y_test[:1]) 


# In[34]:


import pickle
pickle.dump(final_predictions, open('predict.pkl','wb'))


# In[ ]:




