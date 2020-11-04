'''
Module to Prepare The In house forecasts for model.
'''

import os
import calendar
import math
import logging
import datetime

import matplotlib.pyplot as plt
from statsmodels.graphics.correlation import plot_corr
import logger as lg
import pandas as pd
import numpy as np


logger = lg.setup_logger('fp_prep_logger', 'Information_Train.log')
super_logger = lg.setup_logger('second_fp_prep_logger', 'Debug_Train.log',level = logging.DEBUG)



def apply_custgp_code(df, cust_to_group,cust_code_col):
    '''
    Takes the test dataframe along with a dictionary containing customer group for assigned customer code and applies some operations on test data. 
    
    :param df: The test dataframe on which the operations are to be performed.
    :type df: DataFrame

    :param cust_to_group: A dictionary containing customer group for assigned customer code.
    :type cust_to_group: dict
    
    :return: The test dataframe on which few operations have been performed(assigning customer codes for null values).
    :rtype: DataFrame
    '''
    df['Customer Group'] = np.nan
    df[cust_code_col]=pd.to_numeric(df[cust_code_col],errors='coerce')
    df['Customer Group'] = df['Customer Group'].fillna(df[cust_code_col].apply(lambda x: cust_to_group.get(x)))
    
    return df

def pass_year_condition(ib_fcst,pm_fcst,tpms_fcst,ar_partsales_fcst,year_col = 'Year'):
    '''
    Get the Unique year of forecasting
    '''

    if len(ib_fcst[year_col].unique())>1:
        raise ValueError("Please check the unique years of installation for CE forecast (Install base additions")
    else:
        ib_uq_year = ib_fcst[year_col].unique()[0]

    if len(pm_fcst[year_col].unique())>1:
        raise ValueError("Please check the unique years for forecasted PM Deliveries")
    else:
        pm_uq_year = pm_fcst[year_col].unique()[0]

    if len(tpms_fcst[year_col].unique())>1:
        raise ValueError("Please check the unique years for TPMS forecast")
    else:
        tpms_uq_year = tpms_fcst[year_col].unique()[0]
    
    if len(ar_partsales_fcst[year_col].unique())>1:
        raise ValueError("Please check the unique years for forecasted PM Deliveries")
    else:
        arps_uq_year = ar_partsales_fcst[year_col].unique()[0]


    return ib_uq_year == pm_uq_year == tpms_uq_year == arps_uq_year


def get_forecast_year_and_previos_year(any_forecast_df,any_historical_df,year_col = 'Year'):
    '''
    Get the forecasted year and the previous year viz available.

    :param any_forecast_df: Any of the install base , pm delivery or tpms forecast/input 
    :type any_forecast_df: DataFrame

    :param any_historical_df: Any of the install base , pm delivery or tpms historical data
    :type any_historical_df: DataFrame

    :param year_col: Column label for year
    :type year_col: str

    :return: The forecast year and the previous year abailable
    :rtype: int
    '''
    
    forecast_year = any_forecast_df[year_col][0]
    years = sorted(list(any_historical_df[year_col].unique()))

    if forecast_year in years:
        prev_year = years[years.index(forecast_year)-1]
    elif forecast_year not in years:
        years.append(forecast_year)
        prev_year = years[years.index(forecast_year)-1]

    return forecast_year, prev_year



def modify_install_base_additions(ib_additions, top_machines, ibdf_cols, machine_to_eq):
    '''
    Process the input/forecasted CE additions in Install Base  to model standards (EQ Count per Machine).

    :param ib_additions: Forecasted CE additions in Install Base as read from read_csv.
    :type ib_additions: DataFrame

    :param top_machines: The machines available from the historic Install base.
    :type top_machines: array,list like
    
    :param ibdf_cols: The columns present in the historical collated Install Base dataframe.
    :type ibdf_cols: list like

    :param machine_to_eq: The dictionary with values where the column "Equipment Type" is "CE Filling Machines".
    :type machine_to_eq: dict

    :return: Install base additions with dummies on the machine type and Count
    :rtype: DataFrame
    '''
    
    ib_forecast_c = ib_additions.copy()
    ib_forecast_c['Customer Group'] = ib_forecast_c['Customer Group'].fillna("Unassigned")
    tp_forecast_ib = ib_forecast_c.groupby(['Customer Group','Machine System','Month','Year']).agg({'EQ Count':sum}).reset_index()
    tp_forecast_ib['Machine System'] = tp_forecast_ib['Machine System'].apply(lambda x : x if x in machine_to_eq.keys() else "Other")
    tp_forecast_ib['Equipment Type']=tp_forecast_ib['Machine System'].apply(lambda x : machine_to_eq.get(x))
    tp_forecast_ib = tp_forecast_ib[tp_forecast_ib['Equipment Type']=='CE Filling Machines'].reset_index(drop = True)
    tp_forecast_ib.drop(['Equipment Type'],1,inplace = True)
    
    # Make Columns for machine systems
    forecast_ib_dums = pd.get_dummies(top_machines)
    for col in forecast_ib_dums.columns:
        forecast_ib_dums[col].values[:] = 0
    forecast_ib_dums = forecast_ib_dums.iloc[:len(tp_forecast_ib),:]
    forecast_ib_dums['Other'] = 0
    tp_forecast_ib= pd.concat([tp_forecast_ib,forecast_ib_dums],1)


    for i,v in enumerate(tp_forecast_ib['Machine System']):
        if v in tp_forecast_ib.columns:        
            row_value = tp_forecast_ib.loc[i,"EQ Count"]
            tp_forecast_ib.loc[i,v] = row_value
        else:
            row_value = tp_forecast_ib.loc[i,"EQ Count"]
            tp_forecast_ib.loc[i,"Other"] = row_value


    tp_forecast_ib = tp_forecast_ib.drop(['Machine System','Other'],1)
    tp_forecast_ib = pd.concat([tp_forecast_ib.iloc[:,:4],tp_forecast_ib.iloc[:,4:].add_prefix("IB_")],1)
    tp_forecast_ib = tp_forecast_ib.groupby(['Customer Group','Year','Month']).sum().reset_index()
    tp_forecast_ib = tp_forecast_ib[ibdf_cols]
    
    return tp_forecast_ib

    
    
def running_install_base(historical_ib , ib_additions, prev_year, forecast_year):
    
    '''
    Prepares the Expected CE Additions for Install base in a Month,Year format flowing from the previous year.

    :param historical_ib: The Feature Engineered Historical Install Base dataframe.
    :type historical_ib: DataFrame

    :param ib_additions: The test dataframe on which the operations are to be performed.
    :type ib_additions: DataFrame

    
    :param prev_year: The year previous to the forecast year.
    :type prev_year: int,float

    :param forecast_year: The year for which the predictions are to be made.
    :type forecast_year: int,float

    :return: Forecast Year Install Base with expected additions and as flowing from the previous year.
    :rtype: DataFrame
    '''
    
    df = historical_ib.copy()
    df1 = ib_additions.copy()

    prev_year_ib = df[(df['Year']==prev_year) & (df['Month']==12)].reset_index(drop = True)
    
    monthly_forecast_ib = []
    
    for m in range(1,13):
        additions = df1.copy()[df1['Month']<=m]
        additions['Month'] = m
        flowing = prev_year_ib.copy()
        curr = pd.concat([flowing,additions],0,sort= True)
        curr['Year'] = forecast_year
        curr['Month'] = m
        curr = curr.groupby(['Customer Group','Year','Month']).sum().reset_index()
        monthly_forecast_ib.append(curr)
    tp_forecast_ib_fy = pd.concat(monthly_forecast_ib,0)

    return tp_forecast_ib_fy


def modify_forecasted_pmdelivery(pm_fcst):
    '''
    Preprocess the forecast PM delivery file.
    
    :param pm_fcst: Input file for PM delivery
    :type pm_fcst:  DataFrame

    :return: The processed test (packmat_forecast) dataframe.
    :rtype: DataFrame
    '''

    tp_forecast_pm = pm_fcst.copy()
    tp_forecast_pm['Customer Group'] = tp_forecast_pm['Customer Group'].fillna("Unassigned")
    tot_col = tp_forecast_pm.columns[tp_forecast_pm.columns.str.contains("TOTAL")][0]
    tp_forecast_pm = tp_forecast_pm.drop([tot_col, 'Market Name','Sub Category'],1)
    if "Sh" in tp_forecast_pm.columns:
        tp_forecast_pm.drop(['Sh'],1,inplace = True)
    tp_forecast_pm['Size'] = tp_forecast_pm['Size'].astype('object').apply(str).apply(lambda x: x.replace(" ",""))
    tp_forecast_pm['System'] = tp_forecast_pm['System'].apply(lambda x: x.replace(" ",""))
    return tp_forecast_pm
    

def category_names_to_historical(pm_fcst, name_dict,apply_maps = True):
    '''
    The aliases in the PM delivery category that are to be mapped to the model aliases for the category. 
        The matching names are returned as a dict.
    
    :param pm_fcst: The test dataframe.
    :type pm_fcst: DataFrame

    :param name_dict: The dictionary containing the required column names (for the columns being used) i.e 'Category' for PM Delivery Input
    :type name_dict: dict

    :param apply_maps: Wether to apply the derived mappings for the new columns
    :type apply_maps: bool
    
    :return: if apply_maps is true then the modified pm_fcst with correct category names is returned (DataFrame)
        else if apply_maps if flase then the maps of the category names 
        in the input to the historical available is returned(dict)
    :rtype: df/dict
    '''

    categ = name_dict['Category']
    df = pm_fcst.copy()

    forecast_category_to_packmat_category = {}

    historical_names = [
        'LDP (LIQUID DAIRY PRODUCTS)',
        'STILL DRINK (SD)',
        'JUICE & NECTARS (JN)',
        'FOOD',
        'DAIRY ALTERNATIVE',
        'WINE / SPIRITS',
        'DAIRY PRODUCTS (NON-CORE)']

    for i,string_match in enumerate(["LDP","Still","Juice","Food","Dairy Alternative","Wine","Dairy Products"]):
        if len(df[categ][df[categ].str.contains(string_match)].unique()) > 0:
            forecast_category_to_packmat_category[df[categ][df[categ].str.contains(string_match)].unique()[0]] = historical_names[i]

    if apply_maps:
        df[categ] = df[categ].apply(lambda x : forecast_category_to_packmat_category.get(x))
        return df
    else:
        return forecast_category_to_packmat_category
    


def reshape_pm_forecast(pm_fcst, name_dict, top_cats):
    '''
    Takes the PM delivery input, a dictionary containing required column names and the index values of packmat 
    dataframe grouped by columns 'Year' and 'Catsized' wehre the mean of Actual_xThousand_KPK is less than 99.5.
    ----------
    Parameters:

    :param pm_fcst: PM delivery input
    :type pm_fcst: DataFrame

    :param name_dict: The dictionary containing the required column names (for the columns being used).
    :type name_dict: dict
    
    :param top_cats: Category combinations that exist historically in the PM delivery 
    :type top_cats: array,list like
    :return: PM delivery input  with almost all the required operations performed.
    :rtype: DataFrame
    '''

    df = pm_fcst.copy()

    categ = name_dict['Category']

    df = df.melt(id_vars=['Customer Group','Assigned Cust Code',categ,'System','Size','Year'],
    value_name='Actual_xThousand_KPK', var_name='Month')
    df['Month'] = df['Month'].apply(lambda x : x.split("-")[0])
    m_dict = {v: k for k,v in enumerate(calendar.month_abbr)}
    df['Month'] = df['Month'].apply(lambda x: m_dict.get(x))
    df['Actual_xThousand_KPK'] = pd.to_numeric(df['Actual_xThousand_KPK'].apply(str).apply(lambda x: x.replace(",","")), errors='coerce')/1000
    df['Catsized'] = df[categ] + "_" + df['System'] +"_"+ df['Size']
    df = df.groupby(['Customer Group',"Catsized","Year","Month"]).agg({'Actual_xThousand_KPK':sum}).reset_index()
    forecast_pm_dums = pd.get_dummies(top_cats)
    for col in forecast_pm_dums.columns:
        forecast_pm_dums[col].values[:] = 0
    forecast_pm_dums['Otherr'] = 0
    final = pd.concat([df,forecast_pm_dums],1).fillna(0)
    for i,v in enumerate(final['Catsized']):
        if v in final.columns:        
            row_value = final.loc[i,"Actual_xThousand_KPK"]
            final.loc[i,v] = row_value
        else:
            row_value = final.loc[i,"Actual_xThousand_KPK"]
            final.loc[i,"Otherr"] = row_value

    return final

def gather_pm_forecast(pm_fcst, hist_packmatdf):
    '''
    Additional check for the required columns/category combinations from the input pm delivery file.
    
    :param df: PM delivery input data Frame as prepared from reshaping.
    :type df: DataFrame

    :param hist_packmatdf: The historiccal packmatdf dataframe (Feature Engineering product).
    :type hist_packmatdf: DataFrame
    
    :return: The PM delivery input with added information from PACKMAT dataframe.
    :rtype: DataFrame
    '''
    df = pm_fcst.copy()

    df = df[df['Actual_xThousand_KPK']>0].reset_index(drop=True)
    df = df.drop(['Catsized'],1)
    df = df.groupby(['Customer Group','Year','Month']).sum().reset_index().sort_values(['Year','Month']).reset_index(drop = True)
    df = pd.concat([df.iloc[:,:4],df.iloc[:,4:].add_prefix("PM_")],1)
    tp_forecast_pm_fy = df[hist_packmatdf.columns]
    return tp_forecast_pm_fy
    

def get_forecasted_hitrate(hitrate_input,value_col = 'Hit Rate'):
    '''
    Gets  the numeric value of the hit rate as given in the input hit rate forecasted file

    :param hitrate_input: The input hitrate file 
    :type hitrate_input: DataFrame

    :param value_col: The column label containing the hit rate value
    :type value_col: str

    :return: Forecasted Hit Rate
    :rtype: float

    '''

    hitrate = hitrate_input[value_col][0]
    hitrate = pd.to_numeric(hitrate,errors = 'coerce')

    return hitrate
    
def prepare_hitrate_dist(historical_tpdf, new_hr):
    '''
    Finds the distribution of TPMS HIT RATE among Customer Groups based on the historical distributions.

    :param historical_tpdf: The Historical TPMS HIT Rate DataFrame containing 
        Month, Year , 'TP Completed Events (Hits),'Total Potential Events' columns
    :type historical_tpdf: DataFrame

    :param new_hr: The in-house forecasted Hit Rate for the forecasting year.
        Ex: 51.2 for SAM in 2020. The new_hr value in this case will be 51.2
    :type new_hr: float

    :return: Forecasted HIT RATE distributed amidst Customer Groups
    :rtype: DataFrame

    '''

    tpdf = historical_tpdf.copy()

    yearly_tpdf = tpdf.groupby(['Year']).sum().drop(['Hit Rate','Month'],1).reset_index()
    yearly_tpdf['Hit Rate'] = (yearly_tpdf['TP Completed Events (Hits)']/yearly_tpdf['Total Potential Events'])*100
    yearly_tpdf=yearly_tpdf[(yearly_tpdf['Year']>2016)&(yearly_tpdf['Year']!=2020)]
    yearly_tpdf = yearly_tpdf.drop(['Total Potential Events','TP Completed Events (Hits)'],1).reset_index(drop = True)
    #yearly_tpdf['Year'] = yearly_tpdf['Year'].apply(str)
    yearly_tpdf = yearly_tpdf.transpose()
    yearly_tpdf.columns = yearly_tpdf[:1].values.reshape(3,)
    yearly_tpdf = yearly_tpdf.iloc[1:]
    foo = tpdf.groupby(['Customer Group','Year']).sum().drop(['Hit Rate','Month'],1).reset_index()
    foo['Hit Rate'] = (foo['TP Completed Events (Hits)']/foo['Total Potential Events'])*100
    foo=foo[(foo['Year']>2016) & (foo['Year']!=2020)]
    foo = foo.drop(['Total Potential Events','TP Completed Events (Hits)'],1).groupby(['Customer Group','Year']).sum().unstack(1).droplevel(0,1).fillna(0)
    foo.columns.name = None
    mean_hr_pattern = pd.DataFrame(foo.mean(1))
    mean_hr_pattern.columns = ['Hit Rate']
    avg_annual = yearly_tpdf.mean(1).round(2)[0]
    devs = mean_hr_pattern.div(avg_annual)

    return (devs*new_hr).reset_index()
    
def clean_customer_groups_if_only_ib(
        install_base_features,
        packmat_features, 
        tpms_features,
        partsales_features,
        install_base_forecast_features,
        customer_group_col = 'Customer Group'):

    '''
    Clubs the customer groups based on their sole presence in Install base or Packaging Material. Does same clubbing for the inhouse forecast inputs for these data sources.
    The same customer groups are clubbed in the part sales & TPMS.
    
    :param install_base_features: The Feature Engineered Install Base dataframe.
    :type install_base_features: DataFrame
    
    :param packmat_features: The Feature Engineered PACKMAT dataframe.
    :type packmat_features: DataFrame

    :param tpms_features: The Feature Engineered TPMS dataframe.
    :type tpms_features: DataFrame
    
    :param partsales_features: The Feature Engineered PartSales dataframe.
    :type partsales_features: DataFrame
    
    :param install_base_forecast_features: The preprocessed Install Base forecast dataframe.
    :type install_base_forecast_features: DataFrame

    :return: 1. The Feature Engineered Install Base dataframe with preprocessing performed w.r.t. PACKMAT and Install Base dataframes.
        2. The Install Base dataframe with preprocessing performed w.r.t. PACKMAT and Install Base dataframes.
        3. The PartSales dataframe with preprocessing performed w.r.t. PACKMAT and Install Base dataframes.
        4. The PartSales dataframe with preprocessing performed w.r.t. PACKMAT and Install Base dataframes.
    :rtype: list/tuple of DataFrames

    '''

    ibdf = install_base_features.copy()
    packmatdf = packmat_features.copy()
    tpdf = tpms_features.copy()
    psdf = partsales_features.copy()
    tp_forecast_ib_fy = install_base_forecast_features.copy()


    no_packmat_but_ib = []
    no_ib_but_packmat = []

    for c in ibdf[customer_group_col].unique():
        if c not in packmatdf[customer_group_col].unique():
            no_packmat_but_ib.append(c)

    for c in packmatdf[customer_group_col].unique():
        if c not in ibdf[customer_group_col].unique():
            no_ib_but_packmat.append(c)

    # Combine these in install base and forecast of install base 
    ibdf[customer_group_col] = ibdf[customer_group_col].apply(lambda x: x if x not in no_packmat_but_ib else "ONLY_IB")
    ibdf = ibdf.groupby([customer_group_col,'Year','Month']).sum().reset_index()

    tp_forecast_ib_fy[customer_group_col] = tp_forecast_ib_fy[customer_group_col].apply(lambda x: x if x not in no_packmat_but_ib else "ONLY_IB")
    tp_forecast_ib_fy = tp_forecast_ib_fy.groupby(['Customer Group','Year','Month']).sum().reset_index()

    # Combine these in Part Sales
    psdf[customer_group_col] = psdf[customer_group_col].apply(lambda x: x if x not in no_packmat_but_ib else "ONLY_IB")
    psdf = psdf.groupby([customer_group_col,'Year','Month']).sum().reset_index()

    # Combine these in TPDF
    tpdf[customer_group_col] = tpdf[customer_group_col].apply(lambda x: x if x not in no_packmat_but_ib else "ONLY_IB")

    return ibdf, tp_forecast_ib_fy, psdf, tpdf,packmatdf


def export_training_stats(training_set,destination):
    '''
    Saves the correlation matrix, training data description and the training data column names and origin datadataframes in respective CSV's and plots the graphs TetraPak feature Correlation.

    :param training_set: The training set used to train the model with labels
    :type training_set: DataFrame

    :param destination: The target directory to save the stats
    :type destination: any valid path, path like

    '''
    train = training_set.copy()
    tr_schema = pd.DataFrame(train.columns)
    tr_schema.columns = ['Feature']

    sr = []
    for c in tr_schema['Feature'].values:
        if "IB" in c:
            sr.append("Install Base")
        elif "PM" in c:
            sr.append("Packaging Material")
        elif "Hit Rate" in c:
            sr.append("TPMS")
        elif "Sales" in c:
            sr.append("Target Variable")
        else:
            sr.append("Others")

    tr_schema['Source'] = sr
    tr_schema.set_index(tr_schema.index+1,drop = True,inplace = True)

    logger.info("Missing values treated")
    super_logger.info("Missing values treated")
    train = train.fillna(0)

    cols_to_drop = train.std()[train.std() < 0.01].index.values
    train = train.drop(cols_to_drop, axis=1)
    logger.info("Near Zero Variance treatment. Dropping: %s",cols_to_drop)
    super_logger.info("Near Zero Variance treatment. Dropping: %s",cols_to_drop)
    
    stats_df = train.drop(['Year','Month'],1)
    stats = stats_df.describe().transpose().drop(['count'],1)

    corr = stats_df.corr().round(2)

    try:
        corr.to_csv(os.path.join(destination,"Correlations.csv"))
        stats.to_csv(os.path.join(destination,"TrainStats.csv"),index = True)
        tr_schema.to_csv(os.path.join(destination,"TrainSchema.csv"),index = False)
    except NameError as nm:
        print('Could not save stats',nm)
    try:
        # Corr Heatmap export
        fig,ax = plt.subplots(figsize = (24,18))
        ax = plot_corr(corr, xnames=list(corr.columns),ax=ax,normcolor=True,cmap='PiYG')
        plt.xticks(fontsize=12, rotation=60)
        plt.yticks(fontsize=12)
        plt.title('TetraPak Part Sales Model : Feature Correlation | ',datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'))
        ax.tight_layout()
        ax.savefig(os.path.join(destination,"CorrelationsHeatMap.jpg"))
    except Exception as exc:
        print("Unable to save correlation heatmap",exc)


def train_data(
        partsales_features,
        install_base_features,
        packmat_features,
        tpms_features,
        forecast_year,
        destination,
        key = ['Customer Group','Year','Month'],
        export_stats = True):
    '''
    Combines the features generated from  Install Base, PACKMAT and TPMS dataframe as input along with the forecast year and generates the training data to be regressed for PartSales.
    
    :param partsales_features: The Feature Engineered PartSales dataframe.
    :type partsales_features: DataFrame

    :param install_base_features: The Feature Engineered Install Base dataframe.
    :type install_base_features: DataFrame

    :param packmat_features: The Feature Engineered PACKMAT dataframe.
    :type packmat_features: DataFrame

    :param tpms_features: The Feature Engineered TPMS dataframe.
    :type tpms_features: DataFrame

    :param forecast_year: The year for which the predictions are to be made.    
    :type forecast_year: int

    :param key: The keys to merge all the featured data sources
    :type key: list of str, list like

    :return: 1. corr: The correlation matrix for the training dataframe.
        2. stats : A description about various factors like variance, standard deviation etc for the training dataframe.
        3. tr_schema : A dataframe containing the column names of train data and their respective origins, i.e., Install Base, PACKMAT, partsales or TPMS.
        4. y : The target variable. (Part Sales)
        5. X_df : The set of independant features in training data.

    :rtype: DataFrame, Series
    '''

    ibdf = install_base_features.copy()
    packmatdf = packmat_features.copy()
    tpdf = tpms_features.copy()
    psdf = partsales_features.copy()

    temp_1 = pd.merge(left=psdf,right=ibdf,on=key,how='outer')
    temp_2 = pd.merge(left=temp_1,right=packmatdf,on=key,how='outer')
    temp_3 = pd.merge(left=temp_2,right=tpdf,on=key,how='left')
    mdf = temp_3.drop(['EQ Count','Actual_xThousand_KPK','Total Potential Events','TP Completed Events (Hits)'],1)
    mdf['Quarter'] = mdf['Month'].apply(lambda x :math.ceil(x/3.))
    exist_years = sorted(list(mdf['Year'].unique()))

    if forecast_year in exist_years:
        train = mdf[mdf['Year']!=forecast_year]
    elif forecast_year not in exist_years:
        train = mdf

    ## Impute with 0
    train = train.fillna(0)

    # Near Zero Variance
    # Drop columns with std < 0.01 from Training and drop the same from test
    cols_to_drop = train.std()[train.std() < 0.01].index.values
    train = train.drop(cols_to_drop, axis=1)


    logger.info("Available years for training: %s",sorted(list(mdf['Year'].unique())))
    super_logger.info("Available years for training: %s",sorted(list(mdf['Year'].unique())))

    y = train['Sum of Net Sales']
    X_df = train.drop(['Year','Customer Group','Sum of Net Sales'],1)

    logger.info("Training Data Shape: %s",X_df.shape)
    super_logger.info("Training Data Shape: %s",X_df.shape)

    if export_stats:
        export_training_stats(training_set = train , destination = destination)

    return X_df, y


def combine_ar_ml(ar_part_sales , predictions, export_loca = True):

    '''
    Combines the AR Part Sales and the ML predictions to predict the annual numbers.
        The completed quarters value is taken from AR and rest is taken from ML.

    :param ar_part_sales: Input AR Part Sales
    :type ar_part_sales: DataFrame

    :param predictions: Output predictions
    :type predictions: DataFrame

    :return: The final predictions including the AR & the ML forecast
    :rtype: DataFrame
    '''

    further_quarters = ar_part_sales[ar_part_sales['AR']==0]['Quarter'].values
    further_ml_forecast = predictions[predictions['Quarter'].isin(further_quarters)]
    
    ## Combine actuals if there and futher forecast
    a = ar_part_sales[ar_part_sales['AR']>0]
    a.columns = ['Year','Quarter','Part Sales']
    a.insert(value='AR',column ='Source',loc=0)

    b = further_ml_forecast
    b.columns = ['Year','Quarter','Part Sales']
    b.insert(value='ML FORECAST',column ='Source',loc=0)

    annual_out = pd.concat([a,b],0,sort = True)
    annual_out = annual_out[['Year','Quarter','Part Sales','Source']].round(2)

    return annual_out

def export_forecast(result,destination,filename = 'Forecast'):
    '''
    Exports the forecast as a local flat file

    :param result: The resultant prediction
    :type result: DataFrame

    :param destination: The destination folder to save the forecast:
    :type destination: path like

    :param filename: The filename to be saved
    :type filename: str
    
    '''

    simulation_folder = datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
    os.makedirs(os.path.join(destination,simulation_folder),exist_ok=True)

    result.to_csv(os.path.join(destination,simulation_folder,f"{filename}.csv"))
