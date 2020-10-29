# Recommended: download and view the HTML version at:
# https://drive.google.com/file/d/13zb7sjsiGBfg7HLsZUvjyZJDebonLvRG/view?usp=sharing

# Introduction
# ----------------------------------------------------------------------
# Many Chinese folks are interested in investing in the US stock market, 
# but few people have access to an American broker. A good idea would be
# to invest in a mutual fund that tracks the S&P 500... but how do you
# choose one?

# I have picked the 6 largest/accessible S&P 500 index funds that are 
# traded in Chinese Yuan:

# 	096001 大成标普500等权重指数
# 	161125 易方达标普500指数
# 	050025 博时标普500ETF连接A
# 	006075 博时标普500ETF连接C
# 	007721 天弘标普500A
# 	007722 天弘标普500C

# This snippet uses the akshare open-source library to get market data
# and Yahoo Finance for S&P 500 data.
# ----------------------------------------------------------------------

import akshare as ak
import pandas as pd
import numpy as np
import plotly.express as px

# makes data comparable while preserving the trend
def calc(mf):
    hstr_list = []
    norm = mf[0]
    for day in mf:
        n_day = day/norm
        hstr_list.append(n_day)
    return hstr_list

df_spx = pd.read_csv('spx.csv')
df_spx.set_index('Date',inplace=True)
df_spx['SPX'] = calc(df_spx['SPX'])
df_spx.dropna(inplace=True)


# For looking up other funds
# df_funds = ak.fund_em_fund_name()
# df_funds.query('基金代码=="096001"')


funds_list = ['096001','161125','006075','050025','007722','007721']

# Putting the funds into one dataframe
df_list = []

for fd in funds_list:
    df = ak.fund_em_open_fund_info(fund=fd, indicator="单位净值走势")
    df = df.drop(['equityReturn','unitMoney'],axis=1)
    df.rename(columns={'x':'date','y':fd},inplace=True)
    df[fd] = df[fd].astype(float)
    df.set_index('date',inplace=True)
    df_list.append(df)

df_sum = pd.concat(df_list,axis=1,sort=True)
df_sum.dropna(inplace=True)
for fd in funds_list:
    df_sum[fd] = calc(df_sum[fd])


df_sum.index = pd.to_datetime(df_sum.index)
df_spx.index = pd.to_datetime(df_spx.index)

df_total = pd.concat([df_spx,df_sum],axis=1)
df_total.dropna(inplace=True)


# Visualize data (I mean... you can totally skip this, but who 
#                 doesn't like a good old interactive graph?)
fig = px.line(df_total,x=df_total.index,y=df_total.columns)
fig.show()


# Calculates which fund has the highest correlation with S&P 500
for item in funds_list:
    cor = df_total[item].corr(df_total['SPX'])
    print(item, cor)
