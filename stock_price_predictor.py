import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL')
#print(df.head())


df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT']=(df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0


df = df[['Adj. Open','HL_PCT','PCT_CHANGE','Adj. Volume']]

print(df.head())
