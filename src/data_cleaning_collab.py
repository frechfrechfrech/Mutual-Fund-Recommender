import pandas as pd
import numpy as np


def create_office_fund_ratings(net_sales_filepath, zip_lookup_filepath, \
    neter_returns_filepath, min_R4Q_sales = 1000000):
    '''
    Parameters
    ------
    net_sales_filepath: filepath to tab-delimited data dump with columns
        fds_broker_id, zip, fund_id, R4Q sales, R4Q redemptions
    zip_lookup_filepath: filepath to tab-delimited quarterly zip_code
        table with Zip, City, State, MSA
    neter_returns_filepath: filepath to csv with fundid, neter, 1yearreturns

    Returns
    -------
    df_fund_lookup
    df_geo_lookup
    df_ratings

    '''
    # read in data
    df_net_sales = pd.read_csv(net_sales_filepath, sep='\t')
    df_net_sales.rename(columns={'GEO_ID':'ZIP'}, inplace=True)
    df_zip_lookup = pd.read_csv(zip_lookup_filepath, sep='\t', encoding='latin-1')
    df_zip_lookup.rename(columns={'ZIP_CODE':'ZIP'}, inplace=True)
    df_er_returns = pd.read_csv(neter_returns_filepath, header=0, \
                    names=['FUND_ID', 'NET_EXPENSE_RATIO', '1_YEAR_RETURNS'])
    df_er_returns.fillna(0, inplace=True)

    # Calculate net sales, project to ratings, filter to offices that meet threshold
    df_1m, df_ratings = raw_sales_to_net_ratings(df_net_sales, min_R4Q_sales = 1000000)
    print('df_1m, df_ratings created')
    # forget df_net_sales

    lst = [df_net_sales]
    del lst
    print('forgot df_net_sales')

    # Create fund, geographic, and office description lookup tables
    df_fund_lookup = make_fund_lookup_table(df_1m, df_er_returns)
    df_office_desc = make_office_description(df_1m, df_er_returns)
    df_geo_lookup = make_geo_lookup_table(df_1m, df_zip_lookup)

    # save the tables to csvs
    df_office_desc.to_csv('data/df_office_desc.csv', index = False)
    df_fund_lookup.to_csv('data/df_fund_lookup.csv', index = False)
    df_geo_lookup.to_csv('data/df_geo_lookup.csv', index = False)
    df_ratings.to_csv('data/df_ratings.csv', index=False)

    return df_fund_lookup, df_geo_lookup, df_ratings

def raw_sales_to_net_ratings(df_net_sales, min_R4Q_sales = 1000000):
    '''
    Calculate net sales, project to ratings, filter to only offices that
    meet threshold

    Parameters
    ----------

    Return
    -----------
    df_1m: Pandas DataFrame
        original dataframe
        columns = ['FDS_BROKER_ID', 'BROKER_NAME', 'BROAD_FUND_CATEGORY',
        'FUND_CATEGORY', 'FUND_ID', 'ZIP', 'FUND_STANDARD_NAME',
        'GLOBAL_SALES','GLOBAL_REDEMPTIONS', 'NET_SALES', 'BD_ZIP', 'RATING']
    df_ratings: Pandas DataFrame
        Office ratings in format ready to be read into surprise
        columns = ['BD_ZIP', 'FUND_ID', 'RATING']
    '''
    # Create BD+zip unique office ID and calculate net sales
    df_net_sales['NET_SALES'] = df_net_sales['GLOBAL_SALES'] + df_net_sales['GLOBAL_REDEMPTIONS']
    df_net_sales['BD_ZIP']=df_net_sales['FDS_BROKER_ID']+'-'+df_net_sales['ZIP'].map(str)

    # filter to only locations with >1M sales in rolling4Q, location + fund
    # combos with positive global sales in R4Q, No Merrill Lynch
    df_1m = df_net_sales.groupby('BD_ZIP').filter(lambda x : x['GLOBAL_SALES'].sum() > min_R4Q_sales)
    df_1m = df_1m[df_1m['GLOBAL_SALES']>0]
    df_1m = df_1m[df_1m['FDS_BROKER_ID'] != 'LMS28710']

    lst = [df_net_sales]
    del lst
    print('df_1m created and forgot df_net_sales inside ratings function')

    # project net sales onto 1-5 rating scale
    df_1m_classes = pd.DataFrame(df_1m.groupby('BD_ZIP').apply(lambda x: \
                pd.qcut(x.NET_SALES, q=5, labels = False, duplicates='drop')))
    df_1m_classes.index.names=['BD_ZIP', 'idx_og']
    df_1m_classes = pd.DataFrame(df_1m_classes.to_records())
    df_1m_classes.index = df_1m_classes['idx_og']

    print('ratings created')
    #join the ratings with the BD_ZIP and fund_ID
    df_1m['RATING']=df_1m_classes['NET_SALES']+1
    df_1m.dropna(inplace=True)
    df_ratings = df_1m.loc[:, ['BD_ZIP','FUND_ID', 'RATING']]

    return df_1m, df_ratings

def make_geo_lookup_table(df_net_sales, df_zip_lookup):
    '''
    Make a lookup table matching bd+zip to its corresponding geographical
    info (city, state, etc.)

    Inputs
    df_net_sales: table with net sales, trimmed to those that met the 1M thresh
    df_zip_lookup: quarterly zip_code table with Zip, City, State, MSA

    Returns
    df_geo_lookup: dataframe of offices with firm and geographic attributes
        columns = ['BD_ZIP', 'ZIP', 'FDS_BROKER_ID', 'BROKER_NAME', 'CITY',
                'COUNTY','STATE', 'STATE_FULL_NAME', 'MSA']
    '''

    df_geo_lookup = df_net_sales.loc[:,['BD_ZIP', 'ZIP','FDS_BROKER_ID', 'BROKER_NAME']]
    df_geo_lookup.drop_duplicates(inplace=True)
    df_geo_lookup = df_geo_lookup.merge(df_zip_lookup, how = 'left', on='ZIP')

    return df_geo_lookup

def make_fund_lookup_table(df_net_sales, df_er_returns):
    '''
    Make a lookup table matching each fund ID to its Morningstar categorization
    and returns and net expense ratio

    Parameters
    --------
    df_net_sales : Pandas DataFrame
                   table with net sales, trimmed to those that met the 1M thresh
    df_er_returns: Pandas DataFrame


    Returns
    ------
    df_fund_lookup: Pandas DataFrame
            dataframe of funds with fund attributes and Net ER, 1 year returns
            columns = ['FUND_ID', 'FUND_STANDARD_NAME', 'FUND_CATEGORY',
            'BROAD_FUND_CATEGORY','NET_EXPENSE_RATIO', '1_YEAR_RETURNS']
    '''
    df_fund_lookup = df_net_sales.loc[:,['FUND_ID', 'FUND_STANDARD_NAME',\
     'FUND_CATEGORY', 'BROAD_FUND_CATEGORY' ]]
    df_fund_lookup.drop_duplicates(inplace=True)
    df_fund_lookup = df_fund_lookup.merge(df_er_returns, how='left', on='FUND_ID')
    df_fund_lookup['FUND_CATEGORY'] = df_fund_lookup['FUND_CATEGORY'].\
            map(lambda x: x.split('US Fund ')[1]) # clean up fund category names

    return df_fund_lookup


def make_office_description(df_net_sales, df_er_returns):
    '''
    Make lookup table matching offices to their funds
        - sorted descending by rolling4Q sales
        - include information about net expense ratio, 1 year returns

    Parameters
    ---------
    df_net_sales: table with net sales, trimmed to those that met the 1M thresh
    df_er_returns:

    Returns
    df_office_desc: dataframe of offices (BD+ZIP), funds sold at that office,
        fund attributes. Sorted descending by rolling4Q sales
        columns = ['BD_ZIP', 'FUND_ID', 'FUND_STANDARD_NAME', 'FUND_CATEGORY',
                  'BROAD_FUND_CATEGORY','NET_EXPENSE_RATIO', '1_YEAR_RETURNS',
                  'GLOBAL_SALES']
    '''
    df_office_desc = df_net_sales.sort_values(by='NET_SALES', axis=0, ascending=False)
    df_office_desc = df_office_desc.merge(df_er_returns, how='left', on='FUND_ID')
    df_office_desc = df_office_desc.loc[:,['BD_ZIP', 'FUND_ID', 'FUND_STANDARD_NAME',\
                'FUND_CATEGORY', 'BROAD_FUND_CATEGORY','NET_EXPENSE_RATIO', \
                '1_YEAR_RETURNS', 'GLOBAL_SALES']]

    return df_office_desc


if __name__ == '__main__':
    net_sales_filepath = 'data/ALL_CLIENTS_fundid_std_name_bd_zip_redemptions_FY2017.txt'
    neter_returns_filepath = 'data/fund_returns_expense_ratio.csv'
    zip_lookup_filepath = 'data/LMS-Zip Code Reference Table-20180103.txt'

    df_fund_lookup, df_location_lookup, df_ratings = create_office_fund_ratings(\
                net_sales_filepath, zip_lookup_filepath, \
                neter_returns_filepath, min_R4Q_sales = 1000000)

    df_ratings_subset = df_ratings.sample(frac=0.05)
    df_ratings_subset.to_csv('data/df_ratings_subset.csv', index=False)


    ## Find largest offices
    # np.unique(df_1m['BD_ZIP'], return_counts=True)
    # BD_ZIP_ROLLED_UP = df_1m.groupby('BD_ZIP').sum()
    # BD_ZIP_ROLLED_UP.sort(columns = 'GLOBAL_SALES', axis=1, inplace =True)
    # largest_offices = BD_ZIP_ROLLED_UP['GLOBAL_SALES'].sort_values()[-40:]

    # EDA
    # counts = df_1m.groupby('BD_ZIP').count()['FDS_BROKER_ID']
    # counts[np.where(counts>500)].hist()
