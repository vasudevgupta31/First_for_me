'''
This Module contains functions that load data from the local directories.
'''
import os
import ntpath
import calendar

import pandas as pd

import helpers

def get_filepaths(directory,include_only_str=None ):
    """Generates file names in a directory tree by walking the tree either top-down or bottom-up.
        For each directory in the tree rooted at directory top (including top itself),
        it yields a 3-tuple (dirpath, dirnames, filenames).
    :param include_only_str: Only the path for files with the matching strings will be returned.
        Defaults to None i.e. file paths for all files will be returned.
    :type include_only_str: str, optional
    ...
    ...
    :return: file paths of all files matching the criteria
    :rtype: str


    :Example:

        dir = 'root/folder/file.xlsx'

        subdir= 'root/folder2/file2.mov'

        subdir2= 'root/folder3/file3.xlsx'

        get_filepaths(root)

        >> ['root/folder/file.xlsx','root/folder2/file2.mov','root/folder3/file3.xlsx']

        get_filepaths(root,include_only_str=['.xl'])

        >> ['root/folder/file.xlsx','root/folder2/file3.xlsx']
    """
    file_paths = []  # List which will store all of the full filepaths.

    if include_only_str is not None:
        # Walk the tree.
        for root, directories, files in os.walk(directory):
            for filename in files:
                for expr in include_only_str:
                    if expr in filename:
                        # Join the two strings in order to form the full filepath.
                        filepath = os.path.join(root, filename)
                        file_paths.append(filepath)  # Add it to the list.
    else:
        # Walk the tree.
        for root, directories, files in os.walk(directory):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.
    return file_paths  # Self-explanatory.

def read_n_clean(path,file=None,drop_cols=None,skiprows=None,**kwargs):
    """
    Reads and does basic cleanup of excel file.

    :param path: Any valid string path is acceptable.
    :type path: str, path object or file-like object

    :param file: Overwrite filename to keep as column.
        By default the file name is extracted from path.
    :type file: str, optional

    :param drop_cols: Specific columns to drop while reading the file
    :type drop_cols: str, list of str

    :param skiprows: Line numbers to skip (0-indexed) or
        number of lines to skip (int) at the start of the file.
    :type skiprows: list-like, int or callable, optional

    :param **kwargs: kwargs as passed to pandas.read_excel.
    ...
    ...
    :return: Cleaned data frame form the excel file basic cleanup & appended year and month.
    :rtype: DataFrame

    """
    try:
        df = pd.read_excel(path,skiprows = skiprows,**kwargs) # Read File
        df = df.dropna(axis = 0 , how='all') # Remove blank rows
        # Remove rows with > 90% missing values for columns
        df = df[((df.isna().sum(1)/df.shape[1])*100 < 90)]

        if len([col for col in df.columns if 'EQ' in col])==1:
            df.rename(columns = {df.columns[df.columns.str.contains('EQ')][0]:'EQ Count',
                               df.columns[df.columns.str.contains("Cust Cod")][0]:'Assigned Cust Code'},
                               inplace = True)

        elif len([col for col in df.columns if 'KPK' in col])==1:
            df.rename(columns = {df.columns[df.columns.str.contains('KPK')][0]:'Actual_xThousand_KPK',
                               df.columns[df.columns.str.contains("Cust Cod")][0]:'Assigned Cust Code'},
                               inplace = True)

        df = df.loc[~df.apply(lambda row: row.astype(str).str.contains('Result').any(), axis=1)]
        folder = ntpath.basename(ntpath.dirname(path))

        if file is None:
            file = ntpath.basename(path)
            df['File'] = str(file) + str(folder)
        else:
            df['File'] = str(file) + str(folder)

        if 'Year' not in [c.title() for c in df.columns]:
            df['Year'] = file[3:7]
        if 'Month' not in [c.title() for c in df.columns]:
            df['Month'] = file[7:9]

        m_dict = {v: k for k,v in enumerate(calendar.month_name)}

        if drop_cols is not None:
            for c in drop_cols:
                if c in df.columns:
                    df.drop(c,1,inplace = True)
        try:
            df['Month'] = pd.to_numeric(df['Month'])
        except ValueError:
            try:
                df['Month'] = str(file).split(".")[0]
                df['Month'] = df['Month'].apply(lambda x: m_dict.get(x))
                if df['Month'].isna().all():
                    df['Month'] = 'TEXT'
                df['Month'] = pd.to_numeric(df['Month'])
            except:
                print("Unable to Fetch Month From File. Please see that month is available in the file.")

        try:
            df['Year'] = pd.to_numeric(df['Year'])
        except ValueError:
            try:
                df['Year'] = ntpath.basename(ntpath.dirname(path))
                df['Year'] = pd.to_numeric(df['Year'])
            except ValueError:
                print("Unable to Fetch Year From File. Please see that Year is available in the files or as file names.")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns

        if 'Customer Group' in df.columns:
            df['Customer Group'] = pd.to_numeric(df['Customer Group'],errors = 'coerce')
            df['Customer Group'] = df['Customer Group'].fillna('Others')

        return df

    except:
        print("Unable to read",file,".Unsupported format, or corrupt file.")
        return None




def create_cust_codes_map(data):
    '''
    Creates a dictionary of Assigned Customer Codes & Customer Group mappings.

    :param data: Data Frame containing columns 'Assigned Cust Code' 'Customer Group'.
    :type data: DataFrame

    :return: Dictionary of mappings {'Cust Code':'Customer Group'}
    :rtype: dict
    '''

    maps = dict(data.groupby(['Assigned Cust Code','Customer Group']).size().index.unique().values)
    return maps

def assign_customergroups(data,maps):
    '''
    Creates a new dataframe containing an additional Customer Group column.
    :param data: Data frame containing "Assigned Cust Code" column.
    :type data: DataFrame

    :param maps: Customer codes to customer group mappings.
    :type maps: dict

    :return: input data frame with additional customer group column.
    :rtype: DataFrame
    '''

    if isinstance(maps, dict):
        data['Customer Group'] = data['Assigned Cust Code'].apply(lambda x : maps.get(x))
    elif isinstance(maps,pd.DataFrame):
        m = dict(maps.groupby(['Assigned Cust Code','Customer Group']).size().index.unique().values)
        data['Customer Group'] = data['Assigned Cust Code'].apply(lambda x : m.get(x))

    return data



def collate_files(folder_path ,drop_cols = None ,sel_cols=None, verbose = False, messages = True,skiprows=0,df_name = None,**kwargs):

    '''
    Reads and compiles historical files for a data source and returns a single data frame.

    :param folder_path: Path of the folder where install base files (xlsx) are placed.
    :type folder_path: str

    :param drop_cols: The columns to drop in the compilation process. Default = None.
    :type drop_cols: list of str, optional

    :param sel_cols: The columns to be extracted specifically. Defaults to None i.e. all columns will be returned. Default = None.
    :type sel_cols: list of str, optional

    :param verbose: Wether to print the processing of each file. Default = False.
    :type verbose: bool , optional

    :param messages: Wether to print the consolidated messages of number of files and errors. Default = True.
    :type messages:  bool , optional

    :param skiprows: Rows to skip at the beginning (0-indexed). Default = 0.
    :type skiprows: int , optioal

    :param df_name: Name of the data frame to be assigned. Defaults to None. ex: 'install_base','tpms' etc.
    :type df_name: str,optional

    :param **kwargs: kwargs as passed to pandas read_excel.

    :return: Collated Data Frame from various files.
    :retype: DataFrame

    '''
    files = pd.DataFrame()

    for i,file_path in enumerate(get_filepaths(folder_path,include_only_str=['.xl'])):
        if verbose:
            print(i,"Working on",file_path)
        try:
            file_name = ntpath.basename(file_path)
            # Read File
            temp = read_n_clean(path = file_path,file=file_name,drop_cols=drop_cols,skiprows=skiprows,**kwargs)
            # Append to consolidated DF
            files = pd.concat([files,temp],0,sort=True)
        except TypeError:
            print("Please check row for the column names skiprows arg")

    try:
        files = files.sort_values(axis = 0 , by = ['Year','Month']).reset_index(drop = True)
    except KeyError:
        print(f"Unable to sort {df_name}")

    # Collect Messages to print
    if messages:
        print("{n} Files scanned for {dfname}".format(n=len(files['File'].unique()),dfname = df_name))
        files.drop(['File'],1,inplace = True)
    else:
        files.drop(['File'],1,inplace = True)

    files.name = df_name

    return files



def write_collated_to_local(directory,dataframes,formats,verbose = False,**kwargs):
    '''
    Make Local Copies of the collated files from data sources in Derived Data folder.
    Creates directory if not available and permissible.
    
    :param directory: Directory to save the files. If Directory doesnt exist and is creatable then creates.
    :type directory: str 

    :param dataframes: The dataframe/s to be written.
    :type dataframes: list of dataframes
    
    :param formats: The format of the files to be saved in the directory. Currently supported for "CSV" & "PICKLE".
    :type formats: str, list of str
           
    :param verbose: Wether to print the file names and locations while writing.
    :type verbose: bool
    ...
    ...
    :return: The files are written in a folder with the format name.
    :rtype: None
    
    :param **kwargs: kwargs as passed to pandas.to_csv() or pandas.to_pickle()
    
    :Example:
    
    1. single dataframe and single format

    write_collated_to_local(directory = "/Root/Dir/Folder", dataframes = [partsales] , formats = ["CSV"])

    
    2. Multiple dataframes and formats
    
    write_collated_to_local(directory = "/Root/Dir/Folder", dataframes = [tpms,partsales,ib] , formats = ["CSV","PICKLE"])
    
    '''
    try:
        if helpers.is_path_exists_or_creatable(directory):
            os.makedirs(os.path.join(directory,"Derived Data"),exist_ok=True)
            folder = os.path.join(directory,"Derived Data")
            if isinstance(dataframes, pd.DataFrame):
                dataframes = [dataframes]

            if isinstance(formats,str):
                formats = [formats]

            for form in formats:
                if "CSV" in form or "csv" in form:
                    os.makedirs(os.path.join(folder,"CSV"),exist_ok=True)
                    saving_dir = os.path.join(folder,"CSV")
                    for df in dataframes:
                        df.to_csv(os.path.join(saving_dir,"{name}.csv".format(name = df.name)),index = False,**kwargs)
                        if verbose:
                            print(f"{df.name} written in {form} format at {saving_dir}")


                elif "PICKLE" in form or "pickle" in form:
                    os.makedirs(os.path.join(folder,"Pickle"),exist_ok=True)
                    saving_dir = os.path.join(folder,"Pickle")                
                    for df in dataframes:
                        df.to_pickle(os.path.join(saving_dir,"{name}.pickle".format(name = df.name)),**kwargs)
                        if verbose:
                            print(f"{df.name} written in {form} format at {saving_dir}")

                else:
                    print("Only CSV and Pickle are supported. Please check the format name/s.")        
        else:
            print("Unable to write collated data source files locally due to insufficient priveleges or bad names in destination directory")
    except NameError:
        print("The dataframe does not exist")

        return

def col_selector(df, cols):
    """
    Selects desired columns from a dataframe

    :param df: Dataframe to subset
    :type df: DataFrame
    :param cols: List of column names to select
    :type cols: list of str
    ...
    ...
    :return: Subset of the input dataframe with selected columns.
    :rtype: DataFrame
    """
    df = df[cols]
    return df
