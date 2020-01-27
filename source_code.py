#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import csv
import copy 

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import boxcox

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # Data Preparation & Preprocessing

# ### <u> (0) Converting data (from txt to CSV)</u>
# Because it takes long, I do not recommend running this snippet. <br>
# converted.csv, which is the output of the code snippet right below, is already in data folder. 

# In[2]:


with open('data/Medicare_Provider_Util_Payment_PUF_CY2017.txt', newline = '') as source_csv:                                                                                          
    csv_reader = csv.reader(source_csv, delimiter='\t')
    with open("data/converted.csv", 'w') as out_csv:  # converted data will be saved as "data.csv"
        csv_writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for each_line in csv_reader:
            csv_writer.writerow(each_line)


# ### <u> (1) Filtering data by country and entity_code</u>
# I limited the analysis to individual service providers (SPs) in the United States. This prevents unintentional external factors from affecting the analysis.

# In[3]:


data = pd.read_csv("data/converted.csv")
data.shape # Original data has 9,847,443 samples with 26 columns/ The first row is excluded because it is irrelevant


# In[4]:


data.head() 


# In[5]:


# Filtering only service provider in the United States
data = data[data["nppes_provider_country"] == "US"]
# Filtering only individual service provider
data = data[data["nppes_entity_code"] == "I"]
# Dropping duplicates if there are duplicates
data = data.drop_duplicates(subset=None, keep="first", inplace=False).reset_index(drop=True)


# In[6]:


data.shape # Applying two filters above ended up with 9,415,529 samples with 26 columns


# Then I classified categorical columns & numerical columns.
# After that, each column is converted to appropriate data type for further preprocessing.

# In[7]:


categorical = ['npi', 'nppes_provider_last_org_name', 'nppes_provider_first_name',
       'nppes_provider_mi', 'nppes_credentials', 'nppes_provider_gender',
       'nppes_entity_code', 'nppes_provider_street1', 'nppes_provider_street2',
       'nppes_provider_city', 'nppes_provider_state', 'nppes_provider_zip',
       'nppes_provider_country', 'provider_type',
       'medicare_participation_indicator', 'place_of_service', 'hcpcs_code',
       'hcpcs_description', 'hcpcs_drug_indicator']
for cat_column in categorical:
    data[cat_column] = data[cat_column].astype("str")
    
numerical = ['line_srvc_cnt','bene_unique_cnt', 'bene_day_srvc_cnt',
             'average_Medicare_allowed_amt', 'average_submitted_chrg_amt', 
             'average_Medicare_payment_amt', 'average_Medicare_standard_amt']
for num_column in numerical: 
    data[num_column] = data[num_column].astype('float')


# ### <u> (2) Creating a new column "medicare_perc"</u>

# I created one new column for further usage. <br> 
# 
# <b> medicare_perc </b> <br>
# Proportion that (Medicare paid after deductible and coinsurance amounts have been deducted) out of (total charges that the provider submitted for the service). That is, it represents the proportion of Medicare actually covered out of total charges.

# In[8]:


data["medicare_perc"] = data["average_Medicare_standard_amt"]/data["average_submitted_chrg_amt"]
numerical.append("medicare_perc")


# ### <u> (3) Aggregating by npi</u>
# For each npi, I chose the sample that had the least coverage from Medicare (lowest medicare_perc). This is the decision to have one row for each unique npi and ultimately find SPs that are outliers. 

# In[9]:


data = data.loc[data.groupby(["npi"])["medicare_perc"].idxmin()]  
data.shape


# ### <u> (4) Filtering data by state</u>
# To remove the effect of being in different states on my analysis, I decided to choose one state that I would work with. Based on the assumption that higher variablility in "medicare_perc" means the higher likelihood of the existence of outliers, I chose "WI" which is Wisconsin. 

# In[10]:


grouping = "nppes_provider_state"
grouped_by_state = pd.DataFrame()
grouped_by_state["mean"] = data.groupby(grouping)["medicare_perc"].mean()
grouped_by_state["std"] = data.groupby(grouping)["medicare_perc"].std()
grouped_by_state["count"] = data.groupby(grouping)["medicare_perc"].count()
grouped_by_state["std/mean"] = grouped_by_state["std"]/grouped_by_state["mean"]
grouped_by_state.sort_values("std/mean", ascending = False).head(10)


# In[11]:


data = data[data["nppes_provider_state"] == "WI"].reset_index(drop = True)
data.shape # ended up with 20,955 rows and 27 columns 


# In[12]:


data.to_csv("data/filtered.csv", index = False) # filtered data with additional column is saved as "filtered.csv"


# # Exploratory Data Analysis (EDA)

# ### <u> (1) Checking unique values of each column</u>

# In[13]:


data = pd.read_csv("data/filtered.csv")
data.nunique()


# There are 3,931 zip codes in the data set. However, that number is way higher than 709 zip codes that Wisconsin has.  

# In[14]:


data["nppes_provider_zip"].head(10) # The problem is that there are extra numbers following the first 5 digits. 


# In[15]:


data["nppes_provider_zip"] = data["nppes_provider_zip"].astype("str").str[:5] # subsetting first 5 digits
data["nppes_provider_zip"].nunique() # There are now 462 zip codes, which is more reasonable


# ### <u> (2) Checking distribution of each numerical column with box-and-whisker plot </u>

# In[16]:


data[numerical].describe() # understanding the basic distribution of the data


# In[17]:


plt.rcParams['figure.figsize'] = [20, 12]
for pos, val in enumerate(numerical): 
    plt.subplot(3, 3, pos+1)
    data[numerical].boxplot(column = val)
    plt.title(val)
plt.savefig('img/box_and_whisker.png')
plt.show()


# ### <u> (3) Checking correlation between numerical columns </u>

# In[18]:


correlations = data[numerical].corr(method ='pearson') # Pearson correlation
correlations


# In[19]:


# plot correlation matrix
f, ax = plt.subplots(figsize=(12, 6))
heatmap = sns.heatmap(correlations, mask=np.zeros_like(correlations, dtype=np.bool),
            cmap="YlGnBu",
            square=True, ax=ax)
f.savefig("img/heatmap_pearson.png", bbox_inches="tight")


# In[20]:


data[numerical].corr(method ='spearman') # Spearman correlation


# ### <u> (4) Transforming each numerical column</u>

# In[21]:


plt.rcParams['figure.figsize'] = [20, 12]
for pos, val in enumerate(numerical): 
    plt.subplot(3, 3, pos+1)
    data[val].hist()
    plt.title(val)
plt.savefig('img/distribution_before_transformation.png')
plt.show()


# In[22]:


data_new = copy.deepcopy(data)
plt.rcParams['figure.figsize'] = [20, 12]
for pos, val in enumerate(numerical): 
    plt.subplot(3, 3, pos+1)
    data_new[val],lmbda = boxcox(data_new[val]+0.001, lmbda = None) # Box-cox transformation 
    print(val+ "\'s lambda: %0.3f" % lmbda)
    data_new[val].hist()
    plt.title(val)
plt.savefig('img/distribution_after_transformation.png')
plt.show()

data_new[numerical].describe()


# # Clustering (K-means)

# ### <u> 1. Selecting columns for clustering & one-hot encoding</u> 
# I tried to include as many categorical and numerical variables as possible. However, I did not include categorical variables that have so mnay unique values and numerical variables that are highly correlated to each other.

# In[23]:


selected_categorical = ['nppes_provider_gender','nppes_entity_code', 
       'medicare_participation_indicator', 'place_of_service','hcpcs_drug_indicator']
selected_numerical = ["bene_day_srvc_cnt", "bene_unique_cnt", "average_Medicare_allowed_amt"]
features = data_new[selected_categorical + selected_numerical]
for i in selected_categorical:
    dummies = pd.get_dummies(features[i], prefix = i)
    features = pd.concat([features, dummies], axis = 1)
    del features[i]
features.head()


# In[24]:


features.describe()


# ### <u> 2. Normalizing every column</u> 

# In[25]:


scaler = StandardScaler()
dfNorm = scaler.fit_transform(features)
dfNorm = pd.DataFrame(dfNorm)
dfNorm.describe()


# ### <u> 3. Finding appropriate K</u> 

# In[26]:


inertia = []
silh = []
K = range(2,20)
for k in K:
    kmeans = KMeans(n_clusters = k, random_state = 1).fit(dfNorm)
    inertia.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(dfNorm, kmeans.labels_)
    silh.append(silhouette_avg)


# In[27]:


plt.rcParams['figure.figsize'] = [10, 6]
plt.plot(K, inertia, 'bo-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Scree plot')
plt.savefig('img/scree_plot.png')
plt.show() # By looking at the graph, roughly near 7 seems to be a good K. 


# In[28]:


plt.rcParams['figure.figsize'] = [10, 6]
plt.plot(K, silh, 'bo-')
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.title('Silhouette score')
plt.savefig('img/silhouette_score.png')
plt.show() # 6 has the highest silhouette score. 


# ### <u> 4. Clustering with chosen K</u> 

# In[29]:


selected = 6
kmeans = KMeans(n_clusters = selected, random_state = 1).fit(dfNorm)
k_means_labels = kmeans.labels_
for i in range(selected):
    print(i+1)
    print(list(k_means_labels).count(i)) # this represents the number of samples in each cluster 
    print("====")


# In[30]:


data["Cluster"] = k_means_labels+1


# In[31]:


data.to_csv("data/clustered_data.csv")


# In[32]:


data.to_csv('data/clustered_data.txt', index=None, sep=' ', mode='w')


# Analysis on clusters was done through Tableau. Check "cluster analysis.twb" inside the same folder. While it is recommended to use twb file for interactivity, if you do not have Tableau, "cluster analysis.pptx" would provide you the same information and would be enough.
