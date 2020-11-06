#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''This notebook gives the following
1. Nifty 50 chart
2. Nifty Daily returns chart
3. Nifty PE Chart
4. Nifty PE Probability Distribution Curve
5. Nifty PE Statistics'''


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
from scipy.stats import norm
import seaborn as sns

mpl.style.use('default')


df = pd.read_csv('/Users/deepak/Downloads/niftyrawpeval.csv')
df['nifty_returns'] = df['nifty'].pct_change(1)
df.set_index('Date', inplace=True)
df['nifty'].plot(figsize=(14,10), kind='line', use_index=True)
plt.title('Nifty 50 Index')
plt.xlabel('Date')
plt.ylabel('Nifty 50 Value')


# In[3]:


df['max_dd'] = df['nifty'] / df['nifty'].cummax() - 1
df['max_dd'].plot(figsize=(14,10))
plt.title('Nifty 50 Max DD')
plt.axhline(-0.05,0, color='black')
plt.axhline(-0.10,0, color='black')
ylabel = 'Max DD in %'


# In[4]:


df['nifty_returns'].plot(figsize=(14,10))
plt.title('Nifty 50 Daily Returns')


# In[22]:


max_pe = df['P/E'].max()
min_pe = df['P/E'].min()
latest_pe = df['P/E'].values[-1]
latest_pe_date = df['P/E'].tail(1)


plt.figure(figsize=(18, 8))

plt.text(28,0.12, 'Black line is current PE which is ' + str(latest_pe))
plt.text(28,0.115, 'Max PE = ' + str(max_pe))
plt.text(28,0.110, 'Min PE = ' + str(min_pe))

plt.title('Probability Distribution of Nifty50 PE')

sns.distplot(df['P/E'], hist=False, kde=True,
             bins=50, color = 'blue', 
             hist_kws={'edgecolor':'white'},
             kde_kws={'linewidth': 2})
plt.axvline(latest_pe,0, color='black')

plt.text(12.5, 0.11,'@peepalcapital | peepalcapital.com',
         fontsize=10, color='gray',
         ha='center', va='bottom', alpha=0.5)  

data = df['P/E']

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('RdYlBu_r')

# Plot histogram.
n, bins, patches = plt.hist(data, 50, normed=1, color='green')
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

  
    
plt.show()
df['P/E'].describe()


# In[20]:


df[['P/E']].plot(figsize=(18,10))
plt.title('Nifty 50 PE Plot')
plt.text( 500.0, 10,'@peepalcapital | peepalcapital.com',
         fontsize=10, color='gray',
         ha='center', va='bottom', alpha=0.5)

plt.show()


# In[21]:


max_pb = df['P/B'].max()
min_pb = df['P/B'].min()
latest_pb = df['P/B'].values[-1]
latest_pb_date = df['P/B'].tail(1)

plt.figure(figsize=(18, 8))

plt.text(6,0.8, 'Black line is current PB which is ' + str(latest_pb))
plt.text(6,0.75, 'Max PB = ' + str(max_pb))
plt.text(6,0.70, 'Min PB = ' + str(min_pb))

plt.title('Probability Distribution of Nifty50 PB')

sns.distplot(df['P/B'], hist=False, kde=True,
             bins=50, color = 'blue', 
             hist_kws={'edgecolor':'white'},
             kde_kws={'linewidth': 2})
plt.axvline(latest_pb,0, color='black')


data = df['P/B']

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('RdYlBu_r')

# Plot histogram.
n, bins, patches = plt.hist(data, 50, normed=1, color='green')
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))
    
plt.text( 5.5, 0.2,'@peepalcapital | peepalcapital.com',
         fontsize=10, color='gray',
         ha='center', va='bottom', alpha=0.5)

plt.show()




df['P/B'].describe()


# In[ ]:




