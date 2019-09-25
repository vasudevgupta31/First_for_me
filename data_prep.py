import pandas as pd
import numpy as np
import datetime
import os
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pickle

print("Reading Data...")
path = '/Users/vasu/Tensor Dynamics/Demand Forecast/WD'
os.chdir(path)

data = pd.read_csv("kreate.csv")
timestamp = data['Average Values']
data.drop(data.columns[data.columns.str.contains('Average|Solar|trust|Total')],axis=1,inplace=True)
timestamp = pd.to_datetime(timestamp)

### Check Missing values
def missing_zero_values_table(df):
        zero_val = (df == 0.00).astype(int).sum(axis=0)
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
        mz_table = mz_table.rename(
        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
        mz_table['Data Type'] = df.dtypes
        mz_table = mz_table[
            mz_table.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " + str(mz_table.shape[0]) +
              " columns that have missing values.")
#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
        return mz_table

missing_zero_values_table(data)


# Feature extraction from date
print("Extracting features...")

timeobj = pd.DataFrame(timestamp)
timeobj.columns = ['timestamp']
timeobj['timestamp'] = pd.to_datetime(timeobj['timestamp'])
timeobj['month'] = pd.DatetimeIndex(timeobj['timestamp']).month
timeobj['hour'] = pd.DatetimeIndex(timeobj['timestamp']).hour
timeobj['day'] =timeobj['timestamp'].dt.dayofweek + 1
timeobj['weekend']=((pd.DatetimeIndex(timeobj['timestamp']).dayofweek) // 5 == 1).astype(int)
timeobj['date'] = pd.to_datetime(timeobj['timestamp'].dt.date)


def season_of_date(date):
    year = str(date.year)
    seasons = {'spring': pd.date_range(start='21/03/'+year, end='20/06/'+year),
               'summer': pd.date_range(start='21/06/'+year, end='22/09/'+year),
               'autumn': pd.date_range(start='23/09/'+year, end='20/12/'+year)}
    if date in seasons['spring']:
        return 'spring'
    if date in seasons['summer']:
        return 'summer'
    if date in seasons['autumn']:
        return 'autumn'
    else:
        return 'winter'
    
seasons = []
for date in timeobj['timestamp'].dt.date:
    season = season_of_date(date)
    seasons.append(season)

timeobj['season'] = seasons

holidays = pd.read_csv("holidays.csv",encoding = "latin-1").iloc[:,0:4]
holidays['date'] = pd.to_datetime(holidays['date'])

holiday_flag = []
for date in timeobj['date']:
    if date in holidays['date'].to_list():
        holiday_flag.append(1)
    else:
        holiday_flag.append(0)
        
timeobj['holiday'] = holiday_flag

season_hotcoded = pd.get_dummies(timeobj['season'])
timeobj.drop(['date','season'],axis=1,inplace = True)
timeobj = timeobj.join(season_hotcoded)

#timeobj.groupby(['day','weekend']).size().reset_index().rename(columns={0:'count'})  # Check 

data = timeobj.join(data)

# Export
data.to_pickle("featured_data")
data.to_csv("featured_data.csv")


# Add climate data
print("Adding climate variables...")
#data = pd.read_pickle("featured_data")

data['timestamp']= pd.to_datetime(data['timestamp'])
delhi = data[['timestamp','Delhi']]

print("Time range for our dataset is:-")
print("From:",delhi['timestamp'].min())
print("To:",delhi['timestamp'].max())

climate_data = pd.read_csv("VIDP.txt")
climate_data.drop('station',1,inplace=True)
climate_data['valid']=pd.to_datetime(climate_data['valid'])

print("No of timesteps in original data:",delhi.shape[0])
print("No of timesteps in climate data:",climate_data.shape[0])

# generate half hourly sequence to match our load sequence
l = (pd.DataFrame(columns=['NULL'],
                  index=pd.date_range('2017-10-01 00:00', '2018-11-01 00:00',
                                      freq='15T'))
       .between_time('00:00','23:45')
       .index.strftime('%Y-%m-%d %H:%M')
       .tolist()
)

t_df = pd.DataFrame(l)
t_df.rename(columns = {t_df.columns[0]:'valid'},inplace = True)
t_df['valid']=pd.to_datetime(t_df['valid'])

climate_data = pd.merge(t_df,climate_data,on="valid",how="outer")

climate_data['tmpc']=pd.to_numeric(climate_data['tmpc'],errors='coerce')
climate_data['relh']=pd.to_numeric(climate_data['relh'],errors='coerce')
climate_data['sped']=pd.to_numeric(climate_data['sped'],errors='coerce')

climate_data['tmpc']=climate_data['tmpc'].rolling(3,center=True,min_periods=1).mean().apply(lambda x: round(x, 1))
climate_data['relh']=climate_data['relh'].rolling(3,center=True,min_periods=1).mean().apply(lambda x: round(x, 1))
climate_data['sped']=climate_data['sped'].rolling(3,center=True,min_periods=1).mean().apply(lambda x: round(x, 1))

print("No of timesteps in original data:",delhi.shape[0])
print("No of timesteps in climate data:",climate_data.shape[0])

climate_data.rename(columns = {'valid':'timestamp'},inplace=True)
df = pd.merge(delhi,climate_data)

print("The dataset df is ready")

# Export
df.to_csv("delhi_data.csv")
df.to_pickle("delhi_data")