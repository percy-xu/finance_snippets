# Introduction
# Election day is only a few days away... Who will win? Who's next?
# 
# This snippet tries to predict the winner of the 2020 election with the *K-Nearest Neighbor* algorithm.
# 
# Is it going to work? Probably not. Am I still gonna do it? You bet.
# 
# This snippet uses data from:
# - [Presidential Approval Rate](https://www.presidency.ucsb.edu/statistics/data/presidential-job-approval) by *The American Presidency Project, UCSB*
# - [Real GDP](https://fred.stlouisfed.org/series/GDPC1) + [Unemployment](https://fred.stlouisfed.org/series/UNRATE) by *FRED*
# - [Terms of U.S. Presidents](https://www.kaggle.com/harshitagpt/us-presidents) by *HARSHITA GUPTA, Kaggle*
# - [S&P (https://finance.yahoo.com/quote/%5EGSPC/history?period1=1569283200&period2=1603065600&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true) by *Yahoo! Finance*
#
# I also used **Beautiful Soup 4** to collect data from:
# - [U.S. House of Representative Seats](https://history.house.ution/Party-Divisions/Party-Divisions/) *Official Website*
# - [U.S. Senate Seats](https://www.senate.gov/history/partydi.htm) *Official Website*
# 
# (p.s. whoever made the US Senate website should be fired - scarping from there was the most painful thing I've ever done... ouch)

# Import libraries and datasets


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
from plotly import graph_objects as go
from collections import Counter


pd.set_option('mode.chained_assignment', None)


df_spx = pd.read_csv('spx.csv')
df_gdp = pd.read_csv('real gdp.csv')
df_unempymt = pd.read_csv('unemployment.csv')
df_apprv = pd.read_csv('potus approval.csv')
df_senate = pd.read_csv('senate seats.csv')
df_house = pd.read_csv('house seats.csv')
df_potus = pd.read_csv('us presidents.csv')


# # Data Cleaning

# Let's first get all of the election days. We'll just use Nov. 1 here for simplicity.

elct_days = []

for i in range(1948,2024,4):
    t = pd.Timestamp(f'{i}-11-01').date()
    elct_days.append(t)


# Now let's write a function that adds the election days into a dataframe.

def add_elctn_days(df, start, end):
    df[start] = pd.to_datetime(df[start])
    df[end] = pd.to_datetime(df[end])
#     df['election day'] = np.nan
    for index, row in df.iterrows():
        for day in elct_days:
            if row[start]<day<row[end]:
                df.at[index,'election day'] = day
    return df


df_potus = add_elctn_days(df_potus,'start','end')
df_potus.head()


# We can now see the election day each president experienced in his term(s).

# # Getting the Target Variable
# The more important question: what happened after the election day? It will have to be one of these:
# - Re-elected
# - Not re-elected
# - End of 2nd term (cannot be re-elected)

potus_list = list(df_potus['president'])
result = []

for i in range(len(potus_list)):
    # if re-elected
    if i != len(potus_list)-1 and potus_list[i] == potus_list[i+1]:
        result.append('Re-elected')
    # if end of term
    elif i != 0 and potus_list[i] == potus_list[i-1]:
        result.append('End of Term')
    # Not elected
    elif i != len(potus_list)-1 and Counter(potus_list)[potus_list[i]] == 1:
        result.append('Not Re-elected')
    else:
        result.append('NA')
        
df_potus['result'] = result
df_potus.head(10)


# JFK was assasinated and Nixon resigned. We'll have to drop them. RIP Kennedy.
# 
# Those presidents who made the second term are not that useful to us too. Let's drop them too.

df_potus.dropna(inplace=True)
df_potus = df_potus[df_potus.result != 'End of Term']
df_potus.reset_index(drop=True,inplace=True)
df_potus


# We are left with this dataframe where the result (our target variable) being either "Re-elected" or "Not re-elected".

# # Adding Predictors
# Now we just have to add predictors to our model. I used two kinds of predictors:
# - **Economical** (Real GDP, Unemployment, S&P 500)
# - **Political** (Approval Rate, Senate Seats, House of Representative Seats)

# ## Economical Predictors

# #### A simple function that calculates the percentage change between two dates
# We will use this function for GDP, Unemployment and S&P 500

def pctg_change(df, president):
    df_temp = df_potus.query(f"president=='{president}'")
    start = df_temp['start'][df_temp['start'].index[0]]
    end = df_temp['election day'][df_temp['election day'].index[0]]
    
    start_value = df.iloc[df.index.get_loc(start,method='nearest')][0]
    end_value = df.iloc[df.index.get_loc(end,method='nearest')][0]
    delta = round((end_value-start_value)/start_value, 5)
    
    return delta


# ### Real GDP
# Let's try our function on the Real GDP dataset::

df_gdp.head()
:

df_gdp.set_index('DATE',inplace=True)
df_gdp.index = pd.to_datetime(df_gdp.index) # set index as dates so our function can identify it
:

gdp_delta_list = [pctg_change(df_gdp, df_potus['president'][i]) for i in range(len(df_potus))]
df_potus['gdp change'] = gdp_delta_list
df_potus.head()


# ### Unemployment and S&P # We kinda only have to copy/paste here.:

df_unempymt.set_index('DATE',inplace=True)
df_unempymt.index = pd.to_datetime(df_unempymt.index) # set index as dates so our function can identify it
unempymt_delta_list = [pctg_change(df_unempymt, df_potus['president'][i]) for i in range(len(df_potus))]
df_potus['unemployment change'] = unempymt_delta_list

df_spx.drop(columns=['Open','High','Low','Close','Volume'], inplace=True) # keep adj. close only
df_spx.set_index('Date',inplace=True)
df_spx.index = pd.to_datetime(df_spx.index) # set index as dates so our function can identify it
spx_delta_list = [pctg_change(df_spx, df_potus['president'][i]) for i in range(len(df_potus))]
df_potus['spx change'] = spx_delta_list

df_potus.head()


# ## Political Predictors

# ### Public Approval Rate:

df_apprv_avg = df_apprv.groupby('President').mean().reset_index()


# Why not plot it?:

fig = px.bar(df_apprv_avg.sort_values(by=['Approving']), 
             x='President', y='Approving',color='President',
             title='Average Approval Rate for Every POTUS, 1945-2020')
fig.show()


# Add this data to our main dataframe::

df_apprv_avg.drop(columns=['Disapproving','Unsure/NoData'], inplace=True)
df_apprv_avg.rename(columns={'Approving':'avg approval'}, inplace=True)
df_potus = df_potus.join(df_apprv_avg.set_index('President'), on='president')
df_potus.head()


# ### House of Representatives and Senate Seats
# Lastly, let's find out how powerful each president's affiliated party was in the House of Representatives and the Senate on the election day.
# 
# Let's write a function that does this for us::

def calculate_rate(df, president):
    
    for i in range(len(df)):
        df.at[i,'Start'] = pd.to_datetime(str(df.at[i,'Start']))
        df.at[i,'End'] = pd.to_datetime(str(df.at[i,'End']))
    
    df_temp = df_potus.query(f"president=='{president}'")
    elctn_day = pd.to_datetime(df_temp['election day'][df_temp['election day'].index[0]])
    party = 'Democrats' if df_temp['party'][df_temp['party'].index[0]] == 'Democratic' else 'Republicans'
    
    for i in range(len(df)):
        if df.at[i,'Start'] < elctn_day < df.at[i,'End']:
            rate = round(df.at[i,party] / df.at[i,'Total Seats'], 5)
    
    return rate
:

house_rate_list = [calculate_rate(df_house, df_potus['president'][i]) for i in range(len(df_potus))]
df_potus['house rate'] = house_rate_list

senate_rate_list = [calculate_rate(df_senate, df_potus['president'][i]) for i in range(len(df_potus))]
df_potus['senate rate'] = senate_rate_list


# #### Finally, we have all of the predictors ready to go :):

df_potus.head()


# ### What the final dataframe looks like:

df_final = df_potus.drop(columns=['start','end','party','election day'])
df_final = df_final[['president', 
                     'gdp change', 'unemployment change', 'spx change', 
                     'avg approval', 'house rate', 'senate rate',
                     'result']]


# Let's take Trump out and store him in a separate variable.:

trump = df_final[df_final['president']=='Donald Trump']
df_final = df_final[:-1]
df_final


# ## Running the KNN algorithm

# Let's make a dataframe that will be fed into the KNN algorithm.:

df_knn = df_final.drop(columns=['president','result'])


# ### Standardize the data:

scaler = StandardScaler()
scaler.fit(df_knn)
df_knn = pd.DataFrame(scaler.transform(df_knn),columns=df_knn.columns)


# Our dataset has now been converted into standardized numbers that can be interpreted by the KNN algorithm:

df_knn


# ### Spliting Training and Testing Data:

X = df_knn
y = df_final['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# To know the optimal value for the number of neighbors, draw an elbow graph:

error_rate = []

for i in range(1,8):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

fig_elbow = px.line(x=np.arange(1,8),y=error_rate, width=800, height=300)
fig_elbow.show()


# A 33% error rate is a bit too large for my taste... but let's just go with 3

# ## Predicting the 2020 Election:

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn.predict(trump.drop(columns=['president','result']))


# This model tells us President Trump will be ***re-elected*** in the 2020 Election.

# # Limitations

# A lot of improvements can be made to this rudimentary model.
# - **Small sample size**: We only have so many presidents from 1945-2020. This is far smaller than a typical sample used for KNN.
# - **Biased sample**: Most of the presidents in our sample are re-elected. This negatively affect the model's accuracy.
# - **Unaccounted factors**: We only have 6 predictors, which is a bit short. An accurate model will need more.

# # Epilogue
# While this is not a super accurate model, I had a lot of fun making it. We'll know the answer in a few days!
