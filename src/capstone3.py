import pandas as pd
import numpy as np
from surprise import SVD, SVDpp, NMF
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate, train_test_split
import data_cleaning_collab as dcc
from gridsearch import grid_search
import pickle
import time
# from collections import defaultdict



def read_ratings_file_to_surprise(filepath):
    '''
    Convert ratings file to surprise ratings data format

    Parameters: filepath to ratings data in csv format

    Returns: surprise_data : ratings in surprise data format
    '''
    reader = Reader(line_format=u'user item rating',rating_scale=(1, 5), \
    sep=',', skip_lines = 1)
    surprise_data = Dataset.load_from_file(filepath, reader=reader)
    return surprise_data

def gridsearch_and_pkl_best(solver, trainset):



    gs = grid_search(solver)
    gs.fit(trainset)
    best_model = gs.best_estimator['rmse']
    best_model_score = gs.best_score

    pickle.dump(gs, open('src/gs_{}.p'.format(solver.__name__), 'wb'))
    pickle.dump(best_model, open('src/best_{}.p'.format(solver.__name__), 'wb'))

    return best_model_score

def to_dict_file(df, filename):
    '''
    convert dataframe to dict format and output pickled version to data/filename.p
    '''
    df_d = df.to_dict(orient='records')
    pickle.dump(df_d, open('data/{}.p'.format(filename), 'wb'))
    return


def predict_unrated(model, bd_zip, df_office_desc):
    '''
    Predict ratings for unrated funds in an office

    Inputs:
    Model: fit surprise model
    BD_ZIP: string identifier of office
    df_office_desc: lookup table

    Returns:
    List of surprise predictions
    '''
    unique_funds = df_office_desc['FUND_ID'].unique()
    BD_ZIP_rated_funds = df_office_desc[df_office_desc['BD_ZIP'] == bd_zip]['FUND_ID']
    unrated_mask = np.isin(unique_funds, BD_ZIP_rated_funds, invert=True)
    BD_ZIP_unrated_funds = unique_funds[unrated_mask]
    # return BD_ZIP_unrated_funds
    predictions = []
    for fund in BD_ZIP_unrated_funds:
        predictions.append(model.predict(bd_zip, fund))
    return predictions

def get_top_n(predictions, df_fund_lookup, n=10, exclude_predicted_rating = True):
    '''
    Return the top-N recommendation for an office from a set of surprise predictions.

    Parameters
    -------
    predictions : list of surprise-type predictions
                  produced in predict_unrated method
    df_fund_lookup : Pandas DataFrame
                     produced in datacleaningcollab.create_office_fund_ratings
    n : integer
        number of recommendations to get
    exclude_predicted_rating : boolean
        If True (default) predicted rating is dropped from array to enable direct
        reading into web app

    Returns
    -----
    top_n_df : Pandas DataFrame
               DataFrame of top n recommendations for the office along with
               fund category, returns, and net expense ratio
               default_columns: ['FUND_ID', 'FUND_STANDARD_NAME', 'FUND_CATEGORY',
               'BROAD_FUND_CATEGORY','NET_EXPENSE_RATIO', '1_YEAR_RETURNS']
    '''
    top_n = np.array([])
    for uid, iid, true_r, est, _ in predictions:
        top_n = np.append(top_n, [iid, est])
    top_n_df = pd.DataFrame(top_n.reshape(-1,2), columns = ['FUND_ID', 'PREDICTED_RATING'])
    top_n_df.sort_values(by='PREDICTED_RATING', axis=0, inplace=True, ascending=False)
    top_n_df = top_n_df.merge(df_fund_lookup, how='left', on='FUND_ID')

    if exclude_predicted_rating:
        top_n_df.drop('PREDICTED_RATING', axis=1, inplace=True)

    return top_n_df.head(n)


def get_office_description(bd_zip, df_office_desc, exclude_global_sales = True,\
    exclude_BD_ZIP = True):
    '''
    Return the top-N recommendation for an office from a set of surprise predictions.

    Parameters
    -------
    bd_zip : string identifier of office
    df_office_desc : Pandas DataFrame

    exclude_global_sales : boolean
        If True (default) global sales column is dropped from array to enable direct
        reading into web app
    exclude_bd_zip : boolean
        If True (default) bd_zip is dropped from array to enable direct
        reading into web app

    Returns
    -----
    df_office : Pandas DataFrame
               DataFrame of top n funds sold at office in the last year (by
               global saless) along with fund category, returns, and net expense ratio
               default_columns: ['FUND_ID', 'FUND_STANDARD_NAME', 'FUND_CATEGORY',
               'BROAD_FUND_CATEGORY','NET_EXPENSE_RATIO', '1_YEAR_RETURNS']

    '''
    df_office = df_office_desc[df_office_desc['BD_ZIP']==bd_zip]
    if exclude_global_sales:
        df_office.drop('GLOBAL_SALES', axis=1, inplace=True)
    if exclude_BD_ZIP:
        df_office.drop('BD_ZIP', axis=1, inplace=True)

    return df_office.head()

def get_predictions_and_descriptions(bd_zip, model, df_fund_lookup,\
    df_office_desc, n=5, to_dict = None):

    '''
    to_dict: None (default) - shouldn't write to dict
        or value to append to pickle filenames

    '''

    unrated_predictions = predict_unrated(model, bd_zip, df_office_desc)
    top_n = get_top_n(unrated_predictions, df_fund_lookup, n=5)
    office_desc = get_office_description(bd_zip, df_office_desc)

    # write top_n and office_desc to dict and then pickle
    if to_dict != None:
        to_dict_file(top_n, 'top_n_{}'.format(to_dict))
        to_dict_file(office_desc, 'office_desc_{}'.format(to_dict))

    return top_n, office_desc

if __name__ == '__main__':

    # Clean the data
    # net_sales_filepath = 'data/ALL_CLIENTS_fundid_std_name_bd_zip_redemptions_FY2017.txt'
    # neter_returns_filepath = 'data/fund_returns_expense_ratio.csv'
    # zip_lookup_filepath = 'data/LMS-Zip Code Reference Table-20180103.txt'
    #
    # df_fund_lookup, df_location_lookup, df_ratings = dcc.create_office_fund_ratings(\
    #             net_sales_filepath, zip_lookup_filepath, \
    #             neter_returns_filepath, min_R4Q_sales = 1000000)
    #
    # df_ratings_subset = df_ratings.sample(frac=0.05)
    # df_ratings_subset.to_csv('data/df_ratings_subset.csv', index=False)


    ## Use Cross-Validation to find the best solver
    # run model_selection.py

    # # Gridsearch to find the best parameters
    # data = read_ratings_file_to_surprise('data/df_ratings_subset.csv')
    # best_SVD_score = gridsearch_and_pkl_best(SVD, data)
    # print('Best SVD score: {}'.format(best_SVD_score))
    #
    # # fit model with best params on full dataset
    # best_SVD = pickle.load(open("src/best_SVD.p", "rb"))
    # data_full = read_ratings_file_to_surprise('data/df_ratings.csv')
    # start = time.time()
    # best_SVD.fit(data_full.build_full_trainset())
    # time_elapsed = time.time() - start
    # print('time elapsed: {}'.format(time_elapsed))
    # pickle.dump(best_SVD, open('src/best_SVD_fit_full_dataset.p', 'wb'))


    #TIME IT
    # start = time.time()
    # office_1 = 'LMS29880-10036'
    # best_SVD_fit = pickle.load(open('src/best_SVD_fit_full_dataset.p', 'rb'))
    # df_fund_lookup = pd.read_csv('data/df_fund_lookup.csv')
    # df_office_desc = pd.read_csv('data/df_office_desc.csv')
    # load_time = time.time()-start
    # print('load time: {}'.format(load_time))
    #
    # start = time.time()
    # top_n, office_desc = get_predictions_and_descriptions(office_1, best_SVD_fit, \
    #     df_fund_lookup, df_office_desc, n=5)
    # predict_time = time.time()-start
    # print('predict time: {}'.format(predict_time))
    # print('top_n')
    # print(top_n)
    # print('office_desc')
    # print(office_desc)

    #predict unrated for test office
    office_1 = 'LMS29880-10036'
    best_SVD_fit = pickle.load(open('src/best_SVD_fit_full_dataset.p', 'rb'))
    df_fund_lookup = pd.read_csv('data/df_fund_lookup.csv')
    df_office_desc = pd.read_csv('data/df_office_desc.csv')

    top_n, office_desc = get_predictions_and_descriptions(office_1, best_SVD_fit, \
        df_fund_lookup, df_office_desc, n=5, to_dict='alex')





    # def get_all_top(predictions, n=10):
    #     '''Return the top-N recommendation for each user from a set of predictions.
    #
    #     Args:
    #         predictions(list of Prediction objects): The list of predictions, as
    #             returned by the test method of an algorithm.
    #         n(int): The number of recommendation to output for each user. Default
    #             is 10.
    #
    #     Returns:
    #     A dict where keys are user (raw) ids and values are lists of tuples:
    #         [(raw item id, rating estimation), ...] of size n.
    #     '''
    #
    #     # First map the predictions to each user.
    #     top_n = defaultdict(list)
    #     all_preds = []
    #     for uid, iid, true_r, est, _ in predictions:
    #         top_n[uid].append((iid, est))
    #         all_preds.append(est)
    #
    #     # Then sort the predictions for each user and retrieve the k highest ones.
    #     for uid, user_ratings in top_n.items():
    #         user_ratings.sort(key=lambda x: x[1], reverse=True)
    #         top_n[uid] = user_ratings[:n]
    #
    #     return top_n, all_preds
    #
    # # top_dict, all_preds = get_all_top(test_predictions)
