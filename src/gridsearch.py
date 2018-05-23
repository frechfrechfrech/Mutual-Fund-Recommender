from surprise import Dataset
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms.matrix_factorization import SVDpp, NMF, SVD
from surprise import BaselineOnly


def grid_search(surprise_model):

    if type(surprise_model()) == type(SVDpp()):

        param_grid = {'n_factors':[20] , 'n_epochs':[20], 'lr_all':[0.005, 0.007, 0.05, 0.07, 0.5, 0.7, 1.0], 'reg_all':[0.02, 0.05, 0.2, 0.5]}
        gs = GridSearchCV(surprise_model, param_grid, measures=['rmse', 'mae'], cv=3,n_jobs=-1,joblib_verbose=1,refit=True)

    elif type(surprise_model()) == type(SVD()):

        param_grid = {'n_epochs':[20], 'lr_all':[0.005, 0.007, 0.05, 0.07, 0.5, 0.7, 1.0], 'reg_all':[0.02, 0.05, 0.2, 0.5]}
        gs = GridSearchCV(surprise_model, param_grid, measures=['rmse', 'mae'], cv=3,n_jobs=-1,joblib_verbose=1,refit=True)

    elif type(surprise_model()) == type(NMF()):

        param_grid = {'n_epochs':[20], 'reg_pu':[0.02, 0.04, 0.06, 0.08, 0.2], 'reg_qi':[0.02, 0.04, 0.06, 0.08, 0.2]}
        gs = GridSearchCV(surprise_model, param_grid, measures=['rmse', 'mae'], cv=3,n_jobs=-1,joblib_verbose=1,refit=True)

    elif type(surprise_model()) == type(BaselineOnly()):
        param_grid = {'bsl_options': {'method': ['als', 'sgd'], 'reg': [1, 2], 'learning_rate': [0.005, 0.05, 0.5, 1.0]}}
        gs = GridSearchCV(surprise_model, param_grid, measures=['rmse', 'mae'], cv=3,n_jobs=-1,joblib_verbose=1,refit=True)

    return gs




if __name__ == '__main__':

    # Use movielens-100K as test
    data = Dataset.load_builtin('ml-100k')
    test = grid_search(SVD)






#
