#!/usr/bin/env python
# coding: utf-8

# In[6]:


import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


data = pd.read_csv('calls.csv')


# In[8]:


data.info()


# In[9]:


data.user = data.user.astype(object)


# In[10]:


dataf = data.loc[data['direction'] != 'Missed',]
dataf.reset_index(drop=True)


# In[11]:


def validar_numeros(x):
    if (len(x) > 1):
        if(x[1] == '7'):
            return x[1:]
        else:
            return '0'
    else:
        return '0'


# In[12]:


dataf.other = dataf.other.apply(lambda x: validar_numeros(x))


# In[13]:


dataf = dataf.loc[dataf['other'] != '0',]
dataf.reset_index(drop=True, inplace=True)


# In[14]:


import networkx as nx
G_asymmetric = nx.DiGraph()


# In[15]:


for i, nlrow in dataf.iterrows():
    if (nlrow[2] == 'Incoming'):
        G_asymmetric.add_edge(nlrow[1],nlrow[0], weight=nlrow[3])
    else:
        G_asymmetric.add_edge(nlrow[0],nlrow[1], weight=nlrow[3])


# In[16]:


plt.figure(figsize=(8, 6))
nx.draw(G_asymmetric, node_size=11, node_color='red')
plt.title('Red movil', size=15)
plt.show()


# In[ ]:




