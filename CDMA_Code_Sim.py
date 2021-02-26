#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


D = [  [1, -1,  1, -1,  1,  1, -1, -1, 1, -1, 1],  
       [-1, -1,  1,  1,  1, -1, -1, 1, 1, -1, 1], 
       [1,  1, -1, -1, -1,  1,  1, -1, 1, -1, 1],
       [1,  1,  1,  1, -1, -1, -1, -1, 1, -1, 1],
       [1, -1,  1, -1,  1,  1, -1, -1, 1, -1, 1],  
       [-1, -1,  1,  1,  1, -1, -1, 1, 1, -1, 1], 
       [1,  1, -1, -1, -1,  1,  1, -1, 1, -1, 1],
       [1,  1,  1,  1, -1, -1, -1, -1, 1, -1, 1] ]; D = np.array(D); D


# In[3]:


C = [  [-1., -1., -1.,  1., -1., -1.,  1., -1.],
       [-1.,  1., -1., -1., -1.,  1.,  1.,  1.],
       [-1., -1.,  1., -1., -1., -1., -1.,  1.],
       [-1.,  1.,  1.,  1., -1.,  1., -1., -1.],
       [-1., -1., -1.,  1.,  1.,  1., -1.,  1.],
       [-1.,  1., -1., -1.,  1., -1., -1., -1.],
       [-1., -1.,  1., -1.,  1.,  1.,  1., -1.],
       [-1.,  1.,  1.,  1.,  1., -1.,  1.,  1.] ]; C = np.array(C); C


# In[4]:


M = C.shape[1]
N, I = D.shape
RECON = []


# In[5]:


M, N, I, RECON


# In[6]:


G = np.zeros(shape=(I,M))


# In[7]:


for n in range(N):
    Z = np.zeros(shape=(I,M))
    for i in range(I):
        for m in range(M):
            Z[i,m] = D[n,i]*C[n,m]
    G = G + Z


# In[8]:


G


# In[9]:


T = np.reshape(G, (1,-1))


# In[10]:


for n in range(N):
    TOT = np.zeros(shape=(1,I))
    R = np.zeros(shape=(I,M))
    for i in range(I):
        for m in range(M):
            R[i,m] = G[i,m] * C[n,m]
            TOT[0,i] += R[i,m]
    RECON.append(TOT[0]/M)
RECON = np.array(RECON)


# In[11]:


RECON


# In[ ]:




