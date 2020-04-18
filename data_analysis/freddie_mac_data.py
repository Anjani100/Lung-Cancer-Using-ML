#!/usr/bin/env python
# coding: utf-8

# In[1]:


import quandl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')


# In[2]:


def state_list():
    fiddy_states = pd.read_html('https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States')
    states = []
    for abbv in (fiddy_states[0]['Name &postal abbreviation[12]']['Name &postal abbreviation[12].1']):
        states.append(abbv)
    return states


# In[8]:


def quandl_all_states():
    states = state_list()
    main_df = pd.DataFrame()

    for i in states:
        quandl_state_code = 'FMAC/HPI_' + str(i)
        #print(quandl_state_code)
        df = quandl.get(quandl_state_code, authtoken = "KwWzLLZqSKFv4jBtSYr6")
        df.rename(columns = {'NSA Value': str(i)}, inplace = True)
        df = pd.DataFrame(df[i])
        #df = df.pct_change()
        df[i] = (df[i] - df[i][0])/df[i][0] * 100.0
        #print(df.head())

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)
    print(main_df.head())
    main_df.to_pickle('United States.pickle')
quandl_all_states()

def HPI_Benchmark():
    df = quandl.get('FMAC/HPI_USA', authtoken = "KwWzLLZqSKFv4jBtSYr6")
    #print(df.head())
    df.rename(columns = {'NSA Value': 'United States'}, inplace = True)
    
    df['United States'] = (df['United States'] - df['United States'][0]) / df['United States'][0] * 100.0
    #df = df['United States']
    return df['United States']


# In[19]:


fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

HPI_data = pd.read_pickle('United States.pickle')
benchmark = HPI_Benchmark()
HPI_data.plot(ax=ax1)
benchmark.plot(color='k',ax=ax1, linewidth=10)

plt.legend().remove()
plt.show()


# In[20]:


print(HPI_data.corr())
print(HPI_data.corr().describe())


# In[45]:


TX1yr = HPI_data['TX'].rolling(12).std()
print(TX1yr.head())


# In[ ]:




