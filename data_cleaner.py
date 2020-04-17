#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import cv2


# In[47]:


df = pd.read_csv('Support Devices.csv')


# In[48]:


for i in range(len(df['Path'])):
    org_img = "D:\\Dataset\\CheXpert\\Dataset\\" + str(df['Path'][i])
    img = cv2.imread(org_img)
    path = "Support Devices\\img" + str(i) + ".jpg"
    cv2.imwrite(path, img)
    print(i)


# In[ ]:




