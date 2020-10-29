# Introduction

# It's no secret President Trump loves talking about the stock market. 
# But is the stock market actually relevant to his campaign?

# This snippet uses poll data from FiveThirtyEight and stock market data from Yahoo! Finance.

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn import preprocessing
import yfinance as yf


pd.set_option('mode.chained_assignment', None)


# Load the dataset into a data frame


useful_cols = ['display_name','fte_grade','sample_size','created_at','yes','no']
df_app = pd.read_csv('president_approval_polls.csv', usecols=useful_cols)

df_app.rename(columns={'display_name':'Poll',
                       'fte_grade':'Grade',
                       'sample_size':'Sample Size',
                       'created_at':'Date',
                       'yes':'Yes',
                       'no':'No'},
              inplace=True)

df_app.head(10)


# Data Cleaning

# We only want to see the more prestigious polls

def is_bad_grade(g):
    good_grades = ['A+','A','A-','A/B']
    if g not in good_grades:
        g = 'BAD'
    return g

df_app['Grade'] = df_app['Grade'].apply(is_bad_grade)
df_app = df_app[df_app['Grade'] != 'BAD']


# There are probably better ways to weight this, but we'll just use a linear one for simplicity

def weight_rating(grade, approval):
    weights = {'A+':1, 'A':0.975, 'A-':0.95, 'A/B':0.925}
    weighted_rating = round(approval * weights[grade], 4)
    return weighted_rating

df_app['Yes_W'] = df_app.apply(lambda x: weight_rating(x['Grade'], x['Yes']), axis=1)


# Finally, convert dates to pandas timestamps and rearrange chronically

df_app['Date'] = df_app['Date'].apply(lambda x:pd.to_datetime(x).date())

df_app_date = df_app.groupby('Date').mean().drop(columns=['Sample Size','No'])

df_app_date.index = pd.to_datetime(df_app_date.index)


# Let's see what it looks like...

fig_app = px.line(df_app_date, 
                  x=df_app_date.index, 
                  y=df_app_date['Yes_W'].rolling('60d').mean(),
                  title='Approval Rate of President Trump, 60 Days Rolling Average',
                  labels={'x':'Date','y':'Weighted Approval Rate %'})
fig_app.show()


# Now let's get the stock market data using the yfinance library

spx = yf.Ticker("^GSPC")

spx_hist = spx.history(start=df_app_date.index.min(), end=df_app_date.index.max())
spx_hist.drop(columns=['Open','High','Low','Volume','Dividends','Stock Splits'],inplace=True)


# And let's see what the S&P 500 looked like up until now...
fig_spx = px.line(spx_hist, x=spx_hist.index, 
                  y=spx_hist['Close'].rolling('60d').mean(), 
                  title='S&P 500, Close Prices, 60 Days Rolling Average')
fig_spx.show()


# We need to normalize these 2 sets of data in order to compare them. 
# This can be achieved by writing a simple funtion that utilizes the pre-processing function in sklearn.

def normalize(df, col):
    x = df[[col]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled, columns=[f'{col}_N'], index=df.index)
    return df_normalized[f'{col}_N']

app_rate = normalize(df_app_date, col='Yes_W')
spx_chgn = normalize(spx_hist, col='Close')


# Now let's plot them together

trace_app_rate = go.Scatter(x=app_rate.index, y=app_rate.rolling('60d').mean(), name='Approval Rate')
trace_spx_chng = go.Scatter(x=spx_chgn.index, y=spx_chgn.rolling('60d').mean(), name='S&P 500')

data = [trace_app_rate, trace_spx_chng]
layout = go.Layout(title='President Trump Approval Rate and S&P 500, 60 Days Rolling Average', 
                   xaxis={'title':'Date'}, 
                   yaxis={'title':'Normalized Value'})

fig = go.Figure(data=data, layout=layout)
fig.show()


# What can we see here?

# It looks like the approval rate and the S&P 500 index do move together... just not at the same time.

# This makes sense because stock market reacts extremely fast to changes in economical outlooks. 
# It takes some time for the "real" effect to be perceived by us common folks. 
# In our case, this time period is roughly 3 months.

# To help us see better, we can plot another line "SPX Delayed" which shifts the original S&P 500 line 
# to the right to align with changes in the approval rate.

delayed_index = spx_chgn.index + pd.offsets.MonthOffset(3)

trace_spx_delayed = go.Scatter(x=delayed_index, y=spx_chgn.rolling('60d').mean(), name='S&P 500 Delayed')

fig.add_trace(trace_spx_delayed)
fig.update_layout(xaxis_range=[delayed_index.min(),
                               app_rate.index.max()])
fig.show()

# Note: if you are viewing the html version (recommended), 
# you can click on a trace in the legend to hide a line!

# Conclusion

# With the "S&P 500 Delayed" line, we can clearly see the stock market and President Trump's approval rate 
# move together fairly closely. This is especially true after the COVID-19 landed in the US in early 2020, 
# but less so in spring 2019 when the FAANG stocks were subjected to market corrections which had more to do
# with the FED and not President Trump. While this does not mean the stock market can be used as a reliable 
# predictor for Trump's approval because of unaccounted confounding variable, we can at the very least conclude 
# that the President does have a good reason to care about the stock market.