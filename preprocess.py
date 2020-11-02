import pandas as pd

def machine(df1, name_dict):
    '''
    Takes a dataframe along with a dictionary containing required column names and produces a dictionary for the values that satisfy a particular requirement, i.e., where the column "Equipment Type" has the value "CE Filling Machines".

    :param df: The dataframe for which the features are to be generated.
    :type df: DataFrame

    :param name_dict: The dictionary containing the required column names (for the columns being used).
    :type name_dict: dict

    
    :return: Resultant dictionary with values where the column "Equipment Type" is "CE Filling Machines".
    :rtype: dict
    '''

    eq_type = name_dict['Equipment Type']
    m_sys = name_dict['Machine System']
    machine_to_eq = dict(df1[df1[eq_type]=='CE Filling Machines'].groupby([m_sys,eq_type]).size().index.values)
    return machine_to_eq


def preprocess_ib(df,name_dict,sel_machine_type='CE Filling Machines',exclude_patterns='Z Do not use!',drop_user_status='Stopped at Customer'):
    '''
    Takes an install base dataframe along with a dictionary containing required column names and generates required features.
    
    :param df: The collated install base dataframe for which the features are to be generated.
    :type df: DataFrame
    :param name_dict: The dictionary containing the required column names (for the columns being used).
    :type name_dict: dict
    :param sel_machine_type: The type of machines to take from the install base while calculating the forecast. Default = 'CE Filling Machines'
    :type sel_machine_type: str,list of str, optional
    :param exclude_patterns: The alias which can be removed as an error to input files. Default = 'Z Do not use!'
    :type exclude_patterns: str,list of str, optional
    :param drop_user_status: Status of customers which should be removed. Default = 'Stopped at Customer'
    :type drop_user_status: str,list of str, optional
    
    
    
    :return: Resultant dataframe with generated features.
    :rtype: DataFrame
    '''
    df1 = df.copy()
    side = None
    
    if isinstance(sel_machine_type, str):
        sel_machine_type = [sel_machine_type]
    
    if isinstance(exclude_patterns, str):
        exclude_patterns = [exclude_patterns]

    if isinstance(drop_user_status, str):
        drop_user_status = [drop_user_status]

    user_status = name_dict['User Status'] # Get name of User Status field
    eq_type = name_dict['Equipment Type']# Get name of Equipment Type field
    m_sys = name_dict['Machine System']# Get name of Machine System field
    df1 = df1[(~df1[user_status].isin(drop_user_status))& (df1[eq_type].isin(sel_machine_type))&(~df1[m_sys].isin(exclude_patterns))] # Filter business relevant fields from install base
    side = df1[m_sys].value_counts().index.values # Take top machines to create dummies
    df1['Customer Group'] = df1['Customer Group'].fillna("Unassigned") # FIll NA CUSTOMERS GROUPS to "Unassigned"
    df1['Assigned Cust Code'] = df1['Assigned Cust Code'].astype('object') # MAKE DTYPE OBJECT OF CUST CODE
    df1['Construction Year'] = df1['Construction Year'].astype('object') # MAKE DTYPE FOR CONST YEAR AS OBJ
    
    dums = pd.get_dummies(df1[m_sys]) # Make Dummies for Machines
    df1.drop([m_sys],1,inplace = True)
    df1 = pd.concat([df1,dums],1)
    df1 = df1.groupby(['Customer Group','Year','Month']).sum().reset_index().sort_values(axis = 0 , by = ['Year','Month']).reset_index(drop=True)
    df1 = pd.concat([df1.iloc[:,:4],df1.iloc[:,4:].add_prefix("IB_")],1)
    
    return df1,side


def preprocess_pm(df,name_dict,top_perc_combs=99.5):
    '''
    Takes the PM_DELIVERY dataframe along with a dictionary containing required column names and generates required features from combinations.
    
    :param df: The PM delivery dataframe for which the features are to be generated.
    :type df: DataFrame
    :param name_dict: The dictionary containing the required column names (for the columns being used).
    :type name_dict: dict
    :param top_perc_combs: The percentage of unique combinations from [Packaging System + Size Volume + Category] to consider. Remaining i.e. 100-top_perc_combs are lumped as "Others".Default = 99.5 i.e. 99.5 percent unique combinations are kept. Others are clubbed as 'Otherr'.
    :type top_perc_combs: float,optional
    
    :return: Resultant dataframe with generated features. The features are generated in an alias of type PM_Category_PackagingSystem_Volume.
    :rtype: DataFrame
    '''
    df1 = df.copy()
    
    pckg = name_dict['Package System-Size']
    sz_sys = name_dict['Size Volume']
    categ = name_dict['Category']
    
    
    df1[sz_sys] = df1[sz_sys].astype('int')
    try:
        df1['System'] = df1[pckg].apply(lambda x: x.split("-")[0])
    except:
        try:
            df1['System'] = df1[pckg].apply(lambda x: x[:3])
        except:
            df1['System'] = 'Package System Not Available'
            print("Unable to get Packaging System")
    
    df1['System'] = df1['System'].apply(lambda x: x.replace(" ",""))
    df1[sz_sys] = df1[sz_sys].astype(str).apply(lambda x: x.replace(" ",""))
    df1['Catsized'] = df1[categ] + "_" + df1['System'] +"_"+ df1[sz_sys]
    a = df1.groupby(['Year','Catsized']).agg({"Actual_xThousand_KPK":sum}).sort_values(by='Actual_xThousand_KPK',ascending = False)
    a = a.reset_index().groupby(['Catsized']).agg({"Actual_xThousand_KPK":'mean'}).sort_values(by='Actual_xThousand_KPK',ascending = False)
    a['Perc'] = a.div(a.sum())*100
    a['Perc_cumsum']=a['Perc'].cumsum()
    a=a[a['Perc_cumsum']<=top_perc_combs]
    side = a.index.values
    df1['Catsized'] = df1['Catsized'].apply(lambda x: x if x in side else "Otherr")
    df1['Customer Group'] = df1['Customer Group'].fillna("Unassigned")
    df1['Assigned Cust Code'] = df1['Assigned Cust Code'].astype('object')
    df1['Actual_xThousand_KPK'] = df1['Actual_xThousand_KPK'].apply(str).apply(lambda x : x.replace(",","")).astype('float64')
    dums = pd.get_dummies(df1['Catsized'])
    for c in dums:
        dums[c] = dums[c].astype('float64')
    dums.values[dums != 0] = df1['Actual_xThousand_KPK']
    dums = dums.add_prefix("PM_")
    df1 = pd.concat([df1,dums],1)
    df1 = df1.groupby(['Customer Group','Year','Month']).sum().reset_index().sort_values(axis = 0 , by = ['Year','Month']).reset_index(drop=True)
    df1['Year'] = df1['Year'].astype('int')
    df1['Month'] = df1['Month'].astype('int')

    return df1,side


def preprocess_partsales(df,name_dict):
    '''
    Takes the partsales dataframe along with a dictionary containing required column names and generates features.
    
    :param df: The Part Sales dataframe for which the features are to be generated.
    :type df: DataFrame
    :param name_dict: The dictionary containing the required column names (for the columns being used).
    :type name_dict: dict

    :return: Resultant dataframe with generated features i.e. Monthly Part Sales with Customer Groups.
    :rtype: DataFrame
    '''
    
    df1 = df.copy()
    sum_of = name_dict['Sum of Net Sales']
    df1['Customer Group'] = df1['Customer Group'].fillna("Unassigned")
    df1['Assigned Cust Code'] = df1['Assigned Cust Code'].astype('object')
    df1[sum_of] = df1[sum_of].astype('float64')
    df1 = df1.groupby(['Customer Group','Year','Month']).sum().reset_index().sort_values(axis = 0 , by = ['Year','Month']).reset_index(drop=True)
    
    return df1


def preprocess_tpms(df,name_dict,backfill=False,install_base_df = None,backfill_verbose = True):
    '''
    Takes the TPMS online dataframe along with a dictionary containing required column names and generates features.
    
    :param df: The TPMS dataframe for which the features are to be generated.
    :type df: DataFrame
    :param name_dict: The dictionary containing the required column names (for the columns being used).
    :type name_dict: dict
    :param backfill: If TPMS Hit Rate of any Customer Group is incomplete going back in time but the Install Base exists, the Hit Rate is populated for the unavailable history with the first Hit Rate value encountered.
    :type backfill: bool
    :param install_base_df: Install base dataframe (Containing Month, Year & Customer Group Column)to backfill, to be provided only if backfill = True.
    :type install_base_df: DataFrame
    :param backfill_verbose: Wether to print the missing year and the months for which backfilling is done.
    :type backfill_verbose: bool
    
    :return: Resultant dataframe with generated features i.e. Monthly Part Sales with Customer Groups.
    :rtype: DataFrame
    
    '''
    df1 = df.copy()
    potential_events = name_dict['Total Potential Events']
    potential_events = name_dict['TP Completed Events (Hits)']
    
    df1['Customer Group'] = df1['Customer Group'].fillna("Unassigned")
    df1['Assigned Cust Code'] = df1['Assigned Cust Code'].astype('object')
    df1['Asset Description'] = df1['Asset Description'].apply(lambda x:x.split('-')[0])
    df1 = df1.groupby(['Customer Group','Year','Month']).sum().reset_index().drop(['Asset Code'],1).sort_values(axis = 0,by=['Year','Month'])
    
    # Backfill
    if backfill:
        if install_base_df is None: # if None and Backfill raise ValueError
            raise ValueError("Backfill is set to True. Please provide the install base dataframe to backfill")
        elif isinstance(install_base_df, pd.DataFrame) == False:# if provided but not dataframe raise TypeError
            raise TypeError("Please provide a DataFrame for Install Base. Either generated from data loading or preprocessing of Install Base")
        
        elif isinstance(install_base_df, pd.DataFrame):
            if "Year" not in install_base_df.columns or "Month" not in install_base_df.columns or "Customer Group" not in install_base_df.columns:
                raise ValueError("Please check for Customer Group, Year and Month in Install Base provided.") # iF dataframe provided and no Year or Month then raise Value Error
            else:
                ref_df = install_base_df.copy()
                ref_df.columns = [c.lower() for c in ref_df.columns]
                try:
                    ref_df['year'] = ref_df['year'].astype('int')
                    ref_df['month'] = ref_df['month'].astype('int')
                except:
                    raise ValueError("Ambiguous values for Year in Install Base provided.")
                
                cust_gps = df1['Customer Group'].unique()

                            
                try:
                    comp_years = []
                    comp_months = [m for m in range(1,13)]
                    for y in install_base_df['Year'].unique():
                        if y not in df1['Year'].unique():
                            comp_years.append(y)

                    incomp_years = []
                    incomp_months = []


                    for year in install_base_df['Year'].unique():
                        if year in df1['Year'].unique():
                            months_avail = df1[df1['Year']==year]['Month'].unique() # available months for the year
                            months_total = [l for l in range(1,13)]        # All months in an year
                            matching = list(set(months_total) & set(months_avail)) # matching months in th eyear
                            months_avail_for_year = len(matching) # how many months are available
                            not_avail = [month for month in months_total if month not in months_avail] # not available - to be backfilled.

                            if months_avail_for_year<12:
                                incomp_years.append(year)
                                for m in not_avail:
                                    incomp_months.append(m)

                    if backfill_verbose:
                        print(f"TPMS Features : For {comp_years} all months are missing")
                        print(f"TPMS Features : For {incomp_years} {incomp_months} are missing")
                        print("Backfilling for every customer group...")

                    hist = []

                    for c in cust_gps: # iteraate backfill over all customer groups
                        for y in comp_years:
                            for m in comp_months:
                                ys = df1[(df1['Customer Group']==c) & (df1['Month']==m)]['Year'].values.min()
                                c_tot_events = df1[(df1['Customer Group']==c) & (df1['Month']==m)&(df1['Year']==ys)]['TP Completed Events (Hits)'].sum()
                                c_pot_events = df1[(df1['Customer Group']==c) & (df1['Month']==m)&(df1['Year']==ys)]['Total Potential Events'].sum()
                                hist.append([c,y,m,c_pot_events,c_tot_events])

                    incomp_hist = []
                    for c in cust_gps:
                        for y in incomp_years:
                            for m in incomp_months:
                                ys = df1[(df1['Customer Group']==c) & (df1['Month']==m)&(df1['Year']!=2019)]['Year'].values.min()
                                c_tot_events = df1[(df1['Customer Group']==c) & (df1['Month']==m)&(df1['Year']==ys)]['TP Completed Events (Hits)'].sum()
                                c_pot_events = df1[(df1['Customer Group']==c) & (df1['Month']==m)&(df1['Year']==ys)]['Total Potential Events'].sum()
                                incomp_hist.append([c,y,m,c_pot_events,c_tot_events])
                    foo = pd.concat([pd.DataFrame(hist),pd.DataFrame(incomp_hist)],0)
                    foo.columns = df1.columns
                    df1 = pd.concat([foo,df1],0).sort_values(axis = 0 , by = ['Year','Month']).reset_index(drop=True)
                    df1['Hit Rate'] = (df1['TP Completed Events (Hits)']/df1['Total Potential Events'])*100
                    df1['Hit Rate'] = df1['Hit Rate'].fillna(0)
                except:
                    print("Unable to Backfill..")
    return df1

def preprocess(df, name_dict):
    '''
    Takes a dataframe along with a dictionary containing required column names and generates required features like dummy variables.
    This is a high level function to preprocess any of the datasource from IB/PM/PartSales/TPMS. 
    Individual functions to preprocess each data source exist separately and this is a backup. Not used currently.
    

    :param df: The dataframe for which the features are to be generated.
    :type df: Dataframe

    :param name_dict: The dictionary containing the required column names (for the columns being used).
    :type name_dict: dict

    
    :return: Resultant dataframe with generated features.
    :rtype: DataFrame
    '''

    df1 = df.copy()
    side = None
    try:
        user_status = name_dict['User Status']
        eq_type = name_dict['Equipment Type']
        m_sys = name_dict['Machine System']
        df1['Construction Year'] = df1['Construction Year'].astype('object')
        df1 = df1[(df1[user_status]!='Stopped at Customer')&(df1[eq_type]=='CE Filling Machines') & (df1[m_sys]!='Z Do not use!')]
        side = df1[m_sys].value_counts().index.values
        df1['Customer Group'] = df1['Customer Group'].fillna("Unassigned")
        df1['Assigned Cust Code'] = df1['Assigned Cust Code'].astype('object')
        dums = pd.get_dummies(df1[m_sys])
        df1.drop([m_sys],1,inplace = True)
        df1 = pd.concat([df1,dums],1)
        df1 = df1.groupby(['Customer Group','Year','Month']).sum().reset_index().sort_values(axis = 0 , by = ['Year','Month']).reset_index(drop=True)
        df1 = pd.concat([df1.iloc[:,:4],df1.iloc[:,4:].add_prefix("IB_")],1)

    except:
        try:
            pckg = name_dict['Package System-Size']
            sz_sys = name_dict['Size Volume']
            categ = name_dict['Category']
            df1['System'] = df1[pckg].apply(lambda x: x.split("-")[0])
            df1['System'] = df1['System'].apply(lambda x: x.replace(" ",""))
            df1[sz_sys] = df1[sz_sys].astype(str).apply(lambda x: x.replace(" ",""))
            df1['Catsized'] = df1[categ] + "_" + df1['System'] +"_"+ df1[sz_sys]
            a = df1.groupby(['Year','Catsized']).agg({"Actual_xThousand_KPK":sum}).sort_values(by='Actual_xThousand_KPK',ascending = False)
            a = a.reset_index().groupby(['Catsized']).agg({"Actual_xThousand_KPK":'mean'}).sort_values(by='Actual_xThousand_KPK',ascending = False)
            a['Perc'] = a.div(a.sum())*100
            a['Perc_cumsum']=a['Perc'].cumsum()
            a=a[a['Perc_cumsum']<=99.5]
            side = a.index.values
            df1['Catsized'] = df1['Catsized'].apply(lambda x: x if x in side else "Otherr")
            df1['Customer Group'] = df1['Customer Group'].fillna("Unassigned")
            df1['Assigned Cust Code'] = df1['Assigned Cust Code'].astype('object')
            df1['Actual_xThousand_KPK'] = df1['Actual_xThousand_KPK'].apply(str).apply(lambda x : x.replace(",","")).astype('float64')
            dums = pd.get_dummies(df1['Catsized'])
            for c in dums:
                dums[c] = dums[c].astype('float64')
            dums.values[dums != 0] = df1['Actual_xThousand_KPK']
            dums = dums.add_prefix("PM_")
            df1 = pd.concat([df1,dums],1)
            df1 = df1.groupby(['Customer Group','Year','Month']).sum().reset_index().sort_values(axis = 0 , by = ['Year','Month']).reset_index(drop=True)
            df1['Year'] = df1['Year'].astype('int')
            df1['Month'] = df1['Month'].astype('int')
        except:
            try:
                sum_of = name_dict['Sum of Net Sales']
                df1['Customer Group'] = df1['Customer Group'].fillna("Unassigned")
                df1['Assigned Cust Code'] = df1['Assigned Cust Code'].astype('object')
                df1[sum_of] = df1[sum_of].astype('float64')
                df1 = df1.groupby(['Customer Group','Year','Month']).sum().reset_index().sort_values(axis = 0 , by = ['Year','Month']).reset_index(drop=True)
            except:
                try:
                    asset = name_dict['Asset Description']
                    df1['Customer Group'] = df1['Customer Group'].fillna("Unassigned")
                    df1['Assigned Cust Code'] = df1['Assigned Cust Code'].astype('object')
                    df1['Asset Description'] = df1['Asset Description'].apply(lambda x:x.split('-')[0])
                    df1 = df1.groupby(['Customer Group','Year','Month']).sum().reset_index().drop(['Asset Code'],1).sort_values(axis = 0,by=['Year','Month'])
                    cust_gps = df1['Customer Group'].unique()
                    comp_years = [2014,2015]
                    incom_year = [2016]
                    comp_months = range(1,13)
                    incomp_months = range(1,7)
                    hist = []
                    for c in cust_gps:
                        for y in comp_years:
                            for m in comp_months:
                                ys = df1[(df1['Customer Group']==c) & (df1['Month']==m)&(df1['Year']!=2019)]['Year'].values.min()
                                c_tot_events = df1[(df1['Customer Group']==c) & (df1['Month']==m)&(df1['Year']!=2019)&(df1['Year']==ys)]['TP Completed Events (Hits)'].sum()
                                c_pot_events = df1[(df1['Customer Group']==c) & (df1['Month']==m)&(df1['Year']!=2019)&(df1['Year']==ys)]['Total Potential Events'].sum()
                                hist.append([c,y,m,c_pot_events,c_tot_events])

                    incomp_hist = []
                    for c in cust_gps:
                        for y in incom_year:
                            for m in incomp_months:
                                ys = df1[(df1['Customer Group']==c) & (df1['Month']==m)&(df1['Year']!=2019)]['Year'].values.min()
                                c_tot_events = df1[(df1['Customer Group']==c) & (df1['Month']==m)&(df1['Year']!=2019)&(df1['Year']==ys)]['TP Completed Events (Hits)'].sum()
                                c_pot_events = df1[(df1['Customer Group']==c) & (df1['Month']==m)&(df1['Year']!=2019)&(df1['Year']==ys)]['Total Potential Events'].sum()
                                incomp_hist.append([c,y,m,c_pot_events,c_tot_events])
                    foo = pd.concat([pd.DataFrame(hist),pd.DataFrame(incomp_hist)],0)
                    foo.columns = df1.columns
                    df1 = pd.concat([foo,df1],0).sort_values(axis = 0 , by = ['Year','Month']).reset_index(drop=True)
                    df1['Hit Rate'] = (df1['TP Completed Events (Hits)']/df1['Total Potential Events'])*100
                    df1['Hit Rate'] = df1['Hit Rate'].fillna(0)
                except:
                    pass
    out = []
    out.append(df1)
    out.append(side)
    return out
