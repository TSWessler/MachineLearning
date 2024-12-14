'''
Name:
    carEcon

Version:
    wessler
    2024 November 20
    1st version

Description:
    *runs several regression models to predict fuel economy of cars given several features
    *reads the Table in carEcon.txt
    *carEcon data adapted from MATLAB ML course


Used by:
    *NOTHING--this is the code to run

Uses:
    *NOTHING

NOTES:

Response Term (Target/Expected Outcome):
FuelEcon

categorical predictors (features):
Car_Truck
Transmission
Drive
AC
City_Highway

to do:
*optimize NN with standardization
'''

import warnings
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.function_base import logspace
from sklearn import linear_model, svm, tree
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from sklearn.neural_network import MLPRegressor


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
########################################################################################################################


########################################################################################################################
# funcs--data
########################################################################################################################


# ======================================================================================================================
# read in data
# ======================================================================================================================


def get_train_and_test_one_hot(name_of_datafile, name_of_response_term):
    df = pd.read_table(name_of_datafile)

    # do 1-hot
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = one_hot_encoder.fit_transform(df[categorical_columns])
    df_one_hot_terms = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_columns))
    df_one_hot = pd.concat([df, df_one_hot_terms], axis=1)
    df_one_hot = df_one_hot.drop(categorical_columns, axis=1)

    # get xvals, yvals
    xvals = df_one_hot.drop(columns=[name_of_response_term])
    yvals = df_one_hot.loc[:, name_of_response_term]

    # split data
    x_train, x_test, y_train, y_test = train_test_split(xvals, yvals, random_state=0, test_size=0.3, shuffle=True)

    return x_train, y_train, x_test, y_test


# ======================================================================================================================
# scale data
# ======================================================================================================================

def scale_traindata_standard_x(unscaled_vals):
    data_scaler = StandardScaler()
    scaled_vals = data_scaler.fit_transform(unscaled_vals)

    return data_scaler, scaled_vals


def scale_traindata_standard_y(unscaled_vals):
    data_scaler = StandardScaler()
    if isinstance(unscaled_vals, pd.Series):
        temp_vals = unscaled_vals.values.reshape(-1, 1)
        scaled_temp = data_scaler.fit_transform(temp_vals)
        scaled_vals = np.ravel(scaled_temp, order='C')
    else:
        scaled_vals = data_scaler.fit_transform(unscaled_vals)

    return data_scaler, scaled_vals


def scale_testdata_standard(data_scaler, unscaled_vals):
    if isinstance(unscaled_vals, pd.DataFrame):
        unscaled_vals = unscaled_vals.to_numpy()
    mean_of_col = data_scaler.mean_
    var_of_col = data_scaler.var_

    scaled_vals = np.zeros(shape=unscaled_vals.shape)
    num_cols = len(scaled_vals[0])
    num_rows = len(scaled_vals)
    col_id = 0
    while col_id < num_cols:
        row_id = 0
        while row_id < num_rows:
            num = unscaled_vals[row_id][col_id] - mean_of_col[col_id]
            denom = math.sqrt(var_of_col[col_id])
            if abs(denom) < 1e-200:
                if abs(denom) > 1e-200:
                    warnings.warn('divided by zero')
                scaled_vals[row_id][col_id] = 0
            elif math.isnan(denom):
                if not math.isnan(denom):
                    warnings.warn('existence of nan')
                scaled_vals[row_id][col_id] = float('nan')
            else:
                scaled_vals[row_id][col_id] = num / denom
            row_id += 1
        col_id += 1

    return scaled_vals


def unscale_data_standard_y(data_scaler,scaled_vals):
    mean = data_scaler.mean_
    var = data_scaler.var_

    unscaled_vals = []
    for val in scaled_vals:
        unscaled_vals.append(mean + math.sqrt(var) * val)

    unscaled_vals = np.array(unscaled_vals)
    unscaled_vals = unscaled_vals.reshape(len(unscaled_vals), )
    return unscaled_vals


def scale_traindata_minmax_x(unscaled_vals):
    data_scaler = MinMaxScaler()
    scaled_vals = data_scaler.fit_transform(unscaled_vals)

    return data_scaler, scaled_vals


def scale_traindata_minmax_y(unscaled_vals):
    data_scaler = MinMaxScaler()
    if isinstance(unscaled_vals, pd.Series):
        temp_vals = unscaled_vals.values.reshape(-1, 1)
        scaled_temp = data_scaler.fit_transform(temp_vals)
        scaled_vals = np.ravel(scaled_temp, order='C')
    else:
        scaled_vals = data_scaler.fit_transform(unscaled_vals)

    return data_scaler, scaled_vals


def scale_testdata_minmax(data_scaler,unscaled_vals):
    # scaled_val = scaled_min + scaled_range / unscaled_range * (unscaled_val - unscaled_min)
    # scaled_min = 0, scaled_max = 1 => scaled_val =  (unscaled_val - unscaled_min) / unscaled_range
    if isinstance(unscaled_vals, pd.DataFrame):
        unscaled_vals = unscaled_vals.to_numpy()
    min_of_cols = data_scaler.data_min_
    max_of_cols = data_scaler.data_max_

    scaled_vals = np.zeros(shape=unscaled_vals.shape)
    num_cols = len(scaled_vals[0])
    num_rows = len(scaled_vals)
    col_id = 0
    while col_id < num_cols:
        row_id = 0
        while row_id < num_rows:
            num = unscaled_vals[row_id][col_id] - min_of_cols[col_id]
            denom = max_of_cols[col_id] - min_of_cols[col_id]
            if abs(denom) < 1e-200:
                if abs(denom) > 1e-200:
                    warnings.warn('divided by zero')
                scaled_vals[row_id][col_id] = 0
            elif math.isnan(denom):
                if not math.isnan(denom):
                    warnings.warn('existence of nan')
                scaled_vals[row_id][col_id] = float('nan')
            else:
                scaled_vals[row_id][col_id] = num / denom
            row_id += 1
        col_id += 1

    return scaled_vals


def unscale_data_minmax_y(scaler,scaled_vals):
    # unscaled_val = unscaled_min + unscaled_range / scaled_range * (scaled_val - scaled_min)
    # scaled_min = 0, scaled_max = 1 => unscaled_val = unscaled_min + unscaled_range * scaled_val
    unscaled_min = scaler.data_min_
    unscaled_max = scaler.data_max_
    unscaled_range = unscaled_max - unscaled_min

    unscaled_vals = []
    for val in scaled_vals:
        unscaled_vals.append(unscaled_min + unscaled_range * val)

    unscaled_vals = np.array(unscaled_vals)
    unscaled_vals = unscaled_vals.reshape(len(unscaled_vals), )
    return unscaled_vals


########################################################################################################################
# training methods
########################################################################################################################

# ======================================================================================================================
# class for making options for model
# ======================================================================================================================
class ModelOptions:
    def __init__(self):
        self.scale = None
        self.degree = None
        self.criterion = None
        self.max_depth = None
        self.min_samples_leaf = None
        self.min_samples_split = None
        self.splitter = None
        self.ccp_alpha = None
        self.kernel = None
        self.C = None
        self.coef0 = None
        self.epsilon = None
        self.gp_kernel = None
        self.gp_kernel_const = None
        self.length_scale = None
        self.noise_level = None
        self.noise_level_bounds_min = None
        self.noise_level_bounds_max = None
        self.normalize_y = None
        self.hidden_layer_sizes = None
        self.activation = None
        self.solver = None
        self.alpha = None


# ======================================================================================================================
# mini-function--introduce model
# ======================================================================================================================

def introduce_model(model_description):
    print('\n\n====================================================================================================')
    print('Model: ' + model_description)
    print('====================================================================================================\n')
    print('training...\n')


# ======================================================================================================================
# train model (and time training)
# ======================================================================================================================

def train_and_time(mdl, x_train, y_train):
    time_start = time.time()
    mdl.fit(x_train, y_train)
    time_end = time.time()
    elapse_time = time_end - time_start
    time_to_train = convert_sec_to_hr_min_sec(elapse_time)

    return mdl, time_to_train


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# least squares regression optimizer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def optimizer_least_squares(x_train, y_train, x_test, y_test):
    code_run_id = 2

    # linear models
    if code_run_id == 1:

        list_scalings = ['none', 'standardized', 'minmax']
        scale_optim = None
        mse_optim = 1e10

        for scale in list_scalings:
            if scale == 'standardized':
                scaler_x_train, x_train_scaled = scale_traindata_standard_x(x_train)
                x_test_scaled = scale_testdata_standard(scaler_x_train,x_test)
                scaler_y_train, y_train_scaled = scale_traindata_standard_y(y_train)
            elif scale == 'minmax':
                scaler_x_train, x_train_scaled = scale_traindata_minmax_x(x_train)
                x_test_scaled = scale_testdata_minmax(scaler_x_train,x_test)
                scaler_y_train, y_train_scaled = scale_traindata_minmax_y(y_train)
            else:
                x_train_scaled = x_train
                x_test_scaled = x_test
                y_train_scaled = y_train
            mdl = linear_model.LinearRegression()
            mdl, time_to_train = train_and_time(mdl, x_train_scaled, y_train_scaled)
            y_pred_train = mdl.predict(x_train_scaled)
            y_pred_test = mdl.predict(x_test_scaled)
            if scale == 'standardized':
                y_pred_train = unscale_data_standard_y(scaler_y_train, y_pred_train)
                y_pred_test = unscale_data_standard_y(scaler_y_train, y_pred_test)
            elif scale == 'minmax':
                y_pred_train = unscale_data_minmax_y(scaler_y_train, y_pred_train)
                y_pred_test = unscale_data_minmax_y(scaler_y_train, y_pred_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            if mse_test < mse_optim:
                mse_optim = mse_test
                scale_optim = scale
            print("\nScaling: {}\nMSE (train): {}\nMSE (test): {}\n"
                  .format(scale, mse_train, mse_test))

        print("\nBest scaling: {}\nBest mse: {}\n".format(scale_optim, mse_optim))

    # polynomial models
    if code_run_id == 2:

        list_scalings = ['none', 'standardized', 'minmax']
        list_degrees = [2, 3, 4]
        scale_optim = None
        deg_optim = None
        mse_optim = 1e10

        for deg in list_degrees:
            x_poly_train = PolynomialFeatures(degree=deg).fit_transform(x_train)
            x_poly_test = PolynomialFeatures(degree=deg).fit_transform(x_test)
            for scale in list_scalings:
                if scale == 'standardized':
                    scaler_x_train, x_train_scaled = scale_traindata_standard_x(x_poly_train)
                    x_test_scaled = scale_testdata_standard(scaler_x_train, x_poly_test)
                    scaler_y_train, y_train_scaled = scale_traindata_standard_y(y_train)
                elif scale == 'minmax':
                    scaler_x_train, x_train_scaled = scale_traindata_minmax_x(x_poly_train)
                    x_test_scaled = scale_testdata_minmax(scaler_x_train, x_poly_test)
                    scaler_y_train, y_train_scaled = scale_traindata_minmax_y(y_train)
                else:
                    x_train_scaled = x_poly_train
                    x_test_scaled = x_poly_test
                    y_train_scaled = y_train
                mdl = linear_model.LinearRegression()
                mdl, time_to_train = train_and_time(mdl, x_train_scaled, y_train_scaled)
                y_pred_train = mdl.predict(x_train_scaled)
                y_pred_test = mdl.predict(x_test_scaled)
                if scale == 'standardized':
                    y_pred_train = unscale_data_standard_y(scaler_y_train, y_pred_train)
                    y_pred_test = unscale_data_standard_y(scaler_y_train, y_pred_test)
                elif scale == 'minmax':
                    y_pred_train = unscale_data_minmax_y(scaler_y_train, y_pred_train)
                    y_pred_test = unscale_data_minmax_y(scaler_y_train, y_pred_test)
                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_test = mean_squared_error(y_test, y_pred_test)
                if mse_test < mse_optim:
                    mse_optim = mse_test
                    deg_optim = deg
                    scale_optim = scale
                print("\nScaling: {}\nDegree: {}\nMSE (train): {}\nMSE (test): {}\n"
                      .format(scale, deg, mse_train, mse_test))

        print("\nBest scaling: {}\nBest Degree: {}\nBest mse: {}\n".format(scale_optim, deg_optim, mse_optim))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# decision tree optimizer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def optimizer_decision_tree(x_train, y_train, x_test, y_test):
    code_run_id = 2

    if code_run_id == 1:
        params = {'criterion': ['absolute_error', 'squared_error', 'poisson', 'friedman_mse'],
                  'max_depth': [6, 7, 8, 9, 10, 11],
                  'min_samples_leaf': [1, 2, 3, 4],
                  'min_samples_split': [2, 3, 4, 5, 6],
                  'splitter': ['best', 'random']}

        mdl = tree.DecisionTreeRegressor()
        gscv = GridSearchCV(mdl, param_grid=params)
        gscv.fit(x_train, y_train)

        print(gscv.best_estimator_)

        mdl = gscv.best_estimator_
        mdl, time_to_train = train_and_time(mdl, x_train, y_train)
        y_pred_train = mdl.predict(x_train)
        y_pred_test = mdl.predict(x_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        print("MSE (train): {}\nMSE (test): {}\n".format(mse_train, mse_test))

    if code_run_id == 2:
        mdl = tree.DecisionTreeRegressor()
        path = mdl.cost_complexity_pruning_path(x_train, y_train)
        alpha_vals, impurities = path.ccp_alphas, path.impurities

        mse_opt = 1e10
        alpha_opt = None
        for alpha_val in alpha_vals:
            mdl = tree.DecisionTreeRegressor(random_state=0, ccp_alpha=alpha_val, criterion='friedman_mse',
                                             max_depth=7, min_samples_leaf=2, min_samples_split=3, splitter='best')
            mdl, time_to_train = train_and_time(mdl, x_train, y_train)
            y_pred_train = mdl.predict(x_train)
            y_pred_test = mdl.predict(x_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            if mse_test < mse_opt:
                mse_opt = mse_test
                alpha_opt = alpha_val
            print("alpha value: {}\nMSE (train): {}\nMSE (test): {}\n".format(alpha_val, mse_train, mse_test))

        print("Optimum alpha value: {}\nMSE: {}".format(alpha_opt, mse_opt))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# support vector regression optimizer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def optimizer_support_vector_regression(x_train, y_train, x_test, y_test):
    code_run_id = 1

    scaler_x_train, x_train_scaled = scale_traindata_standard_x(x_train)
    x_test_scaled = scale_testdata_standard(scaler_x_train, x_test)
    scaler_y_train, y_train_scaled = scale_traindata_standard_y(y_train)

    # linear models
    if code_run_id == 1:
        params = {
            'kernel': ["linear"],
            'C': np.linspace(8, 10, 11),
            'coef0': np.linspace(0, 1, 11),
            'epsilon': np.logspace(-7, -5, 5)}

    # polynomial models
    elif code_run_id == 2:
        params = {
            'kernel': ["poly"],
            'degree': [2, 3, 4],
            'C': np.linspace(0.1, 1.1, 11),
            'coef0': np.linspace(.5, 1.5, 5),
            'epsilon': np.logspace(-3, 3, 7)}

    # rbf models
    else:
        params = {
            'kernel': ["rbf"],
            'C': np.linspace(8, 10, 11),
            'coef0': np.linspace(0, 1, 11),
            'epsilon': np.logspace(-3, 3, 7)}

    mdl = svm.SVR()
    gscv = GridSearchCV(mdl, param_grid=params)
    gscv, time_to_train = train_and_time(gscv, x_train_scaled, y_train_scaled)

    print('Time to train (hh:mm:ss): '
          + time_to_train.hrs + ":" + time_to_train.mins + ":" + time_to_train.secs + '\n')

    mdl_best = gscv.best_estimator_
    mdl_best, time_to_train = train_and_time(mdl_best, x_train_scaled, y_train_scaled)
    y_pred_train = mdl_best.predict(x_train_scaled)
    y_pred_train = unscale_data_standard_y(scaler_y_train, y_pred_train)
    y_pred_test = mdl_best.predict(x_test_scaled)
    y_pred_test = unscale_data_standard_y(scaler_y_train, y_pred_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    print("best SVR:")
    print(gscv.best_estimator_)
    print('train error: {}\ntest error: {}\n'.format(mse_train, mse_test))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# gaussian process regression optimizer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def optimizer_gaussian_process_regressor(x_train, y_train, x_test, y_test):
    code_run_id = 2
    scale = 'minmax'

    if scale == 'standardized':
        scaler_x_train, x_train_scaled = scale_traindata_standard_x(x_train)
        x_test_scaled = scale_testdata_standard(scaler_x_train, x_test)
        scaler_y_train, y_train_scaled = scale_traindata_standard_y(y_train)
    elif scale == 'minmax':
        scaler_x_train, x_train_scaled = scale_traindata_minmax_x(x_train)
        x_test_scaled = scale_testdata_minmax(scaler_x_train, x_test)
        scaler_y_train, y_train_scaled = scale_traindata_minmax_y(y_train)
    else:
        x_train_scaled = x_train
        x_test_scaled = x_test
        y_train_scaled = y_train

    if code_run_id == 1:  # rbf

        optimum_mse = 1e10
        optimum_param1 = None
        optimum_param2 = None
        optimum_param3 = None

        list_param1 = [True]
        list_param2 = np.logspace(-5, 5, 11)
        list_param3 = np.logspace(-5, 2, 8)

        len_param1 = len(list_param1)
        len_param2 = len(list_param2)
        len_param3 = len(list_param3)
        num_IDs = len_param1 * len_param2 * len_param3

        ID_param1 = 0
        ID_param2 = 0
        ID_param3 = -1
        ID = 0
        while ID < num_IDs:
            ID += 1
            ID_param3 += 1
            if ID_param3 == len_param3:
                ID_param3 = 0
                ID_param2 += 1
                if ID_param2 == len_param2:
                    ID_param2 = 0
                    ID_param1 += 1
            param1 = list_param1[ID_param1]
            param2 = list_param2[ID_param2]
            param3 = list_param3[ID_param3]
            kernel = param2 * RBF(length_scale=param3)
            mdl = GaussianProcessRegressor(kernel=kernel, normalize_y=param1, n_restarts_optimizer=1)
            mdl, time_to_train = train_and_time(mdl, x_train_scaled, y_train_scaled)
            y_pred_train = mdl.predict(x_train_scaled)
            y_pred_test = mdl.predict(x_test_scaled)
            if scale == 'standardized':
                y_pred_train = unscale_data_standard_y(scaler_y_train, y_pred_train)
                y_pred_test = unscale_data_standard_y(scaler_y_train, y_pred_test)
            elif scale == 'minmax':
                y_pred_train = unscale_data_minmax_y(scaler_y_train, y_pred_train)
                y_pred_test = unscale_data_minmax_y(scaler_y_train, y_pred_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            print('Time to train (hh:mm:ss): '
                  + time_to_train.hrs + ":" + time_to_train.mins + ":" + time_to_train.secs)
            print("model: normalize_y={}, disp={}, length_scale={}\nmse (train)={}\nmse (test)={}"
                  .format(param1, param2, param3, mse_train, mse_test))

            if mse_test < optimum_mse:
                optimum_mse = mse_test
                optimum_param1 = param1
                optimum_param2 = param2
                optimum_param3 = param3

        print("\n\n\noptimum model: normalize_y={}, disp={}, length_scale={}\nmse={}\n\n\n"
              .format(optimum_param1, optimum_param2, optimum_param3, optimum_mse))

        kernel_best = optimum_param2 * RBF(length_scale=optimum_param3)
        mdl_best = GaussianProcessRegressor(kernel=kernel_best, normalize_y=optimum_param1, n_restarts_optimizer=1)
        mdl_best, time_to_train = train_and_time(mdl_best, x_train_scaled, y_train_scaled)
        y_pred_train = mdl_best.predict(x_train_scaled)
        y_pred_test = mdl_best.predict(x_test_scaled)
        if scale == 'standardized':
            y_pred_train = unscale_data_standard_y(scaler_y_train, y_pred_train)
            y_pred_test = unscale_data_standard_y(scaler_y_train, y_pred_test)
        elif scale == 'minmax':
            y_pred_train = unscale_data_minmax_y(scaler_y_train, y_pred_train)
            y_pred_test = unscale_data_minmax_y(scaler_y_train, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        print('Time to train (hh:mm:ss): ' + time_to_train.hrs + ":" + time_to_train.mins + ":" + time_to_train.secs)
        print('train error: {}\ntest error: {}\n'.format(mse_train, mse_test))

    if code_run_id == 2:  # constant kernel

        optimum_mse = 1e10
        optimum_param1 = None
        optimum_param2 = None

        list_param1 = [True, False]
        list_param2 = np.logspace(-8, 8, 17)

        len_param1 = len(list_param1)
        len_param2 = len(list_param2)
        num_IDs = len_param1 * len_param2

        ID_param1 = 0
        ID_param2 = -1
        ID = 0
        while ID < num_IDs:
            ID += 1
            ID_param2 += 1
            if ID_param2 == len_param2:
                ID_param2 = 0
                ID_param1 += 1
            param1 = list_param1[ID_param1]
            param2 = list_param2[ID_param2]
            kernel = RBF() + param2
            mdl = GaussianProcessRegressor(kernel=kernel, normalize_y=param1, n_restarts_optimizer=1)
            mdl, time_to_train = train_and_time(mdl, x_train_scaled, y_train_scaled)
            y_pred_train = mdl.predict(x_train_scaled)
            y_pred_test = mdl.predict(x_test_scaled)
            if scale == 'standardized':
                y_pred_train = unscale_data_standard_y(scaler_y_train, y_pred_train)
                y_pred_test = unscale_data_standard_y(scaler_y_train, y_pred_test)
            elif scale == 'minmax':
                y_pred_train = unscale_data_minmax_y(scaler_y_train, y_pred_train)
                y_pred_test = unscale_data_minmax_y(scaler_y_train, y_pred_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            print(
                'Time to train (hh:mm:ss): ' + time_to_train.hrs + ":" + time_to_train.mins + ":" + time_to_train.secs)
            print("model: normalize_y={}, const={}\nmse (train)={}\nmse (test)={}"
                  .format(param1, param2, mse_train, mse_test))

            if mse_test < optimum_mse:
                optimum_mse = mse_test
                optimum_param1 = param1
                optimum_param2 = param2

        print("\n\n\noptimum model: normalize_y={}, const={}\nmse={}\n\n\n"
              .format(optimum_param1, optimum_param2, optimum_mse))

        kernel_best = RBF() + optimum_param2
        mdl_best = GaussianProcessRegressor(kernel=kernel_best, normalize_y=optimum_param1, n_restarts_optimizer=1)
        mdl_best, time_to_train = train_and_time(mdl_best, x_train_scaled, y_train_scaled)
        y_pred_train = mdl_best.predict(x_train_scaled)
        y_pred_test = mdl_best.predict(x_test_scaled)
        if scale == 'standardized':
            y_pred_train = unscale_data_standard_y(scaler_y_train, y_pred_train)
            y_pred_test = unscale_data_standard_y(scaler_y_train, y_pred_test)
        elif scale == 'minmax':
            y_pred_train = unscale_data_minmax_y(scaler_y_train, y_pred_train)
            y_pred_test = unscale_data_minmax_y(scaler_y_train, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        print('Time to train (hh:mm:ss): ' + time_to_train.hrs + ":" + time_to_train.mins + ":" + time_to_train.secs)
        print('train error: {}\ntest error: {}\n'.format(mse_train, mse_test))

    if code_run_id == 3:  # white noise kernel
        time_start_param_search = time.time()
        optimum_mse = 1e10
        optimum_param1 = None
        optimum_param2 = None
        optimum_param3 = None
        optimum_param4 = None

        list_param1 = [True]
        list_param2 = np.logspace(-2, -1, 2)
        list_param3 = np.logspace(-7, -4, 4)
        list_param4 = np.linspace(1, 10, 10)

        len_param1 = len(list_param1)
        len_param2 = len(list_param2)
        len_param3 = len(list_param3)
        len_param4 = len(list_param4)
        num_IDs = len_param1 * len_param2 * len_param3 * len_param4

        ID_param1 = 0
        ID_param2 = 0
        ID_param3 = 0
        ID_param4 = -1
        ID = 0
        while ID < num_IDs:
            ID += 1
            ID_param4 += 1
            if ID_param4 == len_param4:
                ID_param4 = 0
                ID_param3 += 1
                if ID_param3 == len_param3:
                    ID_param3 = 0
                    ID_param2 += 1
                    if ID_param2 == len_param2:
                        ID_param2 = 0
                        ID_param1 += 1
            param1 = list_param1[ID_param1]
            param2 = list_param2[ID_param2]
            param3 = list_param3[ID_param3]
            param4 = list_param4[ID_param4]
            kernel = DotProduct() + WhiteKernel(noise_level=param2, noise_level_bounds=(param3, param4))
            mdl = GaussianProcessRegressor(kernel=kernel, normalize_y=param1, n_restarts_optimizer=10)
            mdl, time_to_train = train_and_time(mdl, x_train_scaled, y_train_scaled)
            y_pred_train = mdl.predict(x_train_scaled)
            y_pred_test = mdl.predict(x_test_scaled)
            if scale == 'standardized':
                y_pred_train = unscale_data_standard_y(scaler_y_train, y_pred_train)
                y_pred_test = unscale_data_standard_y(scaler_y_train, y_pred_test)
            elif scale == 'minmax':
                y_pred_train = unscale_data_minmax_y(scaler_y_train, y_pred_train)
                y_pred_test = unscale_data_minmax_y(scaler_y_train, y_pred_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            print(
                'Time to train (hh:mm:ss): ' + time_to_train.hrs + ":" + time_to_train.mins + ":" + time_to_train.secs)
            print("model: normalize_y={}, noise_level={}, noise_level_bounds=({},{})\nmse (train)={}\nmse (test)"
                  .format(param1, param2,param3, param4, mse_train, mse_test))

            if mse_test < optimum_mse:
                optimum_mse = mse_test
                optimum_param1 = param1
                optimum_param2 = param2
                optimum_param3 = param3
                optimum_param4 = param4

        time_end_param_search = time.time()
        elapse_time = time_end_param_search - time_start_param_search
        time_to_sweep = convert_sec_to_hr_min_sec(elapse_time)
        print(
            '\n\n\nTime to sweep (hh:mm:ss): ' + time_to_sweep.hrs + ":" + time_to_sweep.mins + ":" + time_to_sweep.secs)
        print("param combos tested: {}".format(num_IDs))
        print("optimum model: normalize_y={}, noise_level={}, noise_level_bounds=({},{})\noptimum mse={}\n\n\n"
              .format(optimum_param1, optimum_param2, optimum_param3, optimum_param4, optimum_mse))

        kernel_best = DotProduct() + WhiteKernel(noise_level=optimum_param2,
                                                 noise_level_bounds=(optimum_param3, optimum_param4))
        mdl_best = GaussianProcessRegressor(kernel=kernel_best, normalize_y=optimum_param1, n_restarts_optimizer=1)
        mdl_best, time_to_train = train_and_time(mdl_best, x_train_scaled, y_train_scaled)
        y_pred_train = mdl_best.predict(x_train_scaled)
        y_pred_test = mdl_best.predict(x_test_scaled)
        if scale == 'standardized':
            y_pred_train = unscale_data_standard_y(scaler_y_train, y_pred_train)
            y_pred_test = unscale_data_standard_y(scaler_y_train, y_pred_test)
        elif scale == 'minmax':
            y_pred_train = unscale_data_minmax_y(scaler_y_train, y_pred_train)
            y_pred_test = unscale_data_minmax_y(scaler_y_train, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        print('Time to train (hh:mm:ss): ' + time_to_train.hrs + ":" + time_to_train.mins + ":" + time_to_train.secs)
        print('train error: {}\ntest error: {}\n'.format(mse_train, mse_test))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# multi-layer perceptron (NN) regressor optimizer
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def optimizer_neural_network(x_train, y_train, x_test, y_test):
    list_hidden_layer_sizes = [(10, 10, 10, 100), (10, 10, 50, 100), (10, 10, 100, 100), (10, 10, 500, 100),
                               (10, 10, 1000, 100),
                               (10, 50, 10, 100), (10, 100, 10, 100), (10, 500, 10, 100), (10, 1000, 10, 100),
                               (50, 10, 10, 100), (100, 10, 10, 100), (500, 10, 10, 100), (1000, 10, 10, 100),
                               (50, 50, 10, 100), (50, 50, 50, 100), (50, 50, 100, 100), (50, 50, 500, 100),
                               (50, 50, 1000, 100),
                               (50, 10, 50, 100), (50, 100, 50, 100), (50, 500, 50, 100), (50, 1000, 50, 100),
                               (10, 50, 50, 100), (100, 50, 50, 100), (500, 50, 50, 100), (1000, 50, 50, 100),
                               (100, 100, 10, 100), (100, 100, 50, 100), (100, 100, 100, 100), (100, 100, 500, 100),
                               (100, 100, 1000, 100),
                               (100, 10, 100, 100), (100, 50, 100, 100), (100, 500, 100, 100), (100, 1000, 100, 100),
                               (10, 100, 100, 100), (50, 100, 100, 100), (500, 100, 100, 100), (1000, 100, 100, 100),
                               (500, 500, 10, 100), (500, 500, 50, 100), (500, 500, 100, 100), (500, 500, 500, 100),
                               (500, 500, 1000, 100),
                               (500, 10, 500, 100), (500, 50, 500, 100), (500, 100, 500, 100), (500, 1000, 500, 100),
                               (10, 500, 500, 100), (50, 500, 500, 100), (100, 500, 500, 100), (1000, 500, 500, 100),
                               (1000, 1000, 10, 100), (1000, 1000, 50, 100), (1000, 1000, 100, 100),
                               (1000, 1000, 500, 100), (1000, 1000, 1000, 100),
                               (1000, 10, 1000, 100), (1000, 50, 1000, 100), (1000, 100, 1000, 100),
                               (1000, 500, 1000, 100),
                               (10, 1000, 1000, 100), (50, 1000, 1000, 100), (100, 1000, 1000, 100),
                               (500, 1000, 1000, 100)
                               ]
    list_activation = ['identity', 'logistic', 'tanh', 'relu']
    list_solver = ['lbfgs', 'adam']
    list_alpha = logspace(-5, -1, 5)

    optim_hidden_layer_sizes = None
    optim_activation = None
    optim_solver = None
    optim_alpha = None
    optim_mse = 1e10

    for hidden_layer_sizes in list_hidden_layer_sizes:
        for activation in list_activation:
            for solver in list_solver:
                for alpha in list_alpha:
                    mdl = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                                       alpha=alpha, random_state=0, early_stopping=True)
                    mdl, time_to_train = train_and_time(mdl, x_train, y_train)
                    y_pred = mdl.predict(x_test)
                    mse = mean_squared_error(y_test, y_pred)
                    print('Time to train (hh:mm:ss): '
                          + time_to_train.hrs + ":" + time_to_train.mins + ":" + time_to_train.secs)
                    print("model: hidden_layer_sizes={}, activation={}, solver={}, alpha={}\nmse={}\n\n".format(
                        hidden_layer_sizes, activation, solver, alpha, mse))
                    if mse < optim_mse:
                        optim_mse = mse
                        optim_hidden_layer_sizes = hidden_layer_sizes
                        optim_activation = activation
                        optim_solver = solver
                        optim_alpha = alpha

    print("\n\n\noptimum model:  hidden_layer_sizes={}, activation={}, solver={}, alpha={}\noptimum mse={}\n\n\n"
          .format(optim_hidden_layer_sizes, optim_activation, optim_solver, optim_alpha, optim_mse))

    mdl_best = MLPRegressor(hidden_layer_sizes=optim_hidden_layer_sizes, activation=optim_activation,
                            solver=optim_solver, alpha=optim_alpha)
    mdl_best, time_to_train = train_and_time(mdl_best, x_train, y_train)
    y_pred_train = mdl_best.predict(x_train)
    y_pred_test = mdl_best.predict(x_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print('Time to train (hh:mm:ss): ' + time_to_train.hrs + ":" + time_to_train.mins + ":" + time_to_train.secs)
    print('train error: {}\ntest error: {}\n'.format(mse_train, mse_test))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# least squares linear regression
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_mdl_least_squares(x_train, y_train, x_test, mdl_opts):
    if mdl_opts.degree is not None:
        x_train = PolynomialFeatures(degree=mdl_opts.degree).fit_transform(x_train)
        x_test = PolynomialFeatures(degree=mdl_opts.degree).fit_transform(x_test)
    if mdl_opts.scale == 'standardized':
        scaler_x_train, x_train = scale_traindata_standard_x(x_train)
        x_test = scale_testdata_standard(scaler_x_train, x_test)
        scaler_y_train, y_train = scale_traindata_standard_y(y_train)
    elif mdl_opts.scale == 'minmax':
        scaler_x_train, x_train = scale_traindata_minmax_x(x_train)
        x_test = scale_testdata_minmax(scaler_x_train, x_test)
        scaler_y_train, y_train = scale_traindata_minmax_y(y_train)
    mdl = linear_model.LinearRegression()
    mdl, time_to_train = train_and_time(mdl, x_train, y_train)
    y_pred = mdl.predict(x_test)
    if mdl_opts.scale == 'standardized':
        y_pred = pd.Series(unscale_data_standard_y(scaler_y_train, y_pred))
    elif mdl_opts.scale == 'minmax':
        y_pred = pd.Series(unscale_data_minmax_y(scaler_y_train, y_pred))

    return mdl, y_pred, time_to_train


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# regression tree
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_mdl_decision_tree_regressor(x_train, y_train, x_test, mdl_opts):
    if mdl_opts.kernel is None:
        mdl = tree.DecisionTreeRegressor()
    else:
        mdl = tree.DecisionTreeRegressor(criterion=mdl_opts.criterion, max_depth=mdl_opts.max_depth,
                                         ccp_alpha=mdl_opts.ccp_alpha,
                                         min_samples_leaf=mdl_opts.min_samples_leaf,
                                         min_samples_split=mdl_opts.min_samples_split, splitter=mdl_opts.splitter)
    mdl, time_to_train = train_and_time(mdl, x_train, y_train)
    y_pred = mdl.predict(x_test)

    return mdl, y_pred, time_to_train


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# support vector regression
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_mdl_support_vector_regression(x_train, y_train, x_test, mdl_opts):
    if mdl_opts.scale == 'standardized':
        scaler_x_train, x_train = scale_traindata_standard_x(x_train)
        x_test = scale_testdata_standard(scaler_x_train, x_test)
        scaler_y_train, y_train = scale_traindata_standard_y(y_train)
    elif mdl_opts.scale == 'minmax':
        scaler_x_train, x_train = scale_traindata_minmax_x(x_train)
        x_test = scale_testdata_minmax(scaler_x_train, x_test)
        scaler_y_train, y_train = scale_traindata_minmax_y(y_train)
    if mdl_opts.kernel is None:
        mdl = svm.SVR()
    elif mdl_opts.kernel == 'poly':
        mdl = svm.SVR(kernel=mdl_opts.kernel, C=mdl_opts.C, coef0=mdl_opts.coef0, epsilon=mdl_opts.epsilon,
                      degree=mdl_opts.degree)
    else:
        mdl = svm.SVR(kernel=mdl_opts.kernel, C=mdl_opts.C, coef0=mdl_opts.coef0, epsilon=mdl_opts.epsilon)
    mdl, time_to_train = train_and_time(mdl, x_train, y_train)
    y_pred = mdl.predict(x_test)
    if mdl_opts.scale == 'standardized':
        y_pred = pd.Series(unscale_data_standard_y(scaler_y_train, y_pred))
    elif mdl_opts.scale == 'minmax':
        y_pred = pd.Series(unscale_data_minmax_y(scaler_y_train, y_pred))

    return mdl, y_pred, time_to_train


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# gaussian process regression
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_mdl_gaussian_process_regression(x_train, y_train, x_test, mdl_opts):
    if mdl_opts.scale == 'standardized':
        scaler_x_train, x_train = scale_traindata_standard_x(x_train)
        x_test = scale_testdata_standard(scaler_x_train, x_test)
        scaler_y_train, y_train = scale_traindata_standard_y(y_train)
    elif mdl_opts.scale == 'minmax':
        scaler_x_train, x_train = scale_traindata_minmax_x(x_train)
        x_test = scale_testdata_minmax(scaler_x_train, x_test)
        scaler_y_train, y_train = scale_traindata_minmax_y(y_train)
    if mdl_opts.kernel is None:
        mdl = GaussianProcessRegressor()
    elif mdl_opts.kernel == 'constant':
        kernel = RBF() + mdl_opts.gp_kernel_const
        mdl = GaussianProcessRegressor(kernel=kernel, normalize_y=mdl_opts.normalize_y)
    elif mdl_opts.kernel == 'rbf':
        kernel = mdl_opts.gp_kernel_const * RBF(length_scale=mdl_opts.length_scale)
        mdl = GaussianProcessRegressor(kernel=kernel, normalize_y=mdl_opts.normalize_y)
    else:
        kernel = DotProduct() + WhiteKernel(noise_level=mdl_opts.noise_level, noise_level_bounds=(mdl_opts.noise_level_bounds_min, mdl_opts.noise_level_bounds_max))
        mdl = GaussianProcessRegressor(kernel=kernel, normalize_y=mdl_opts.normalize_y)
    mdl, time_to_train = train_and_time(mdl, x_train, y_train)
    y_pred = mdl.predict(x_test)
    if mdl_opts.scale == 'standardized':
        y_pred = unscale_data_standard_y(scaler_y_train, y_pred)
    elif mdl_opts.scale == 'minmax':
        y_pred = unscale_data_minmax_y(scaler_y_train, y_pred)

    return mdl, y_pred, time_to_train


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# neural network
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_mdl_neural_network(x_train, y_train, x_test, mdl_opts):
    if mdl_opts.solver is None:
        mdl = MLPRegressor()
    else:
        mdl = MLPRegressor(hidden_layer_sizes=mdl_opts.hidden_layer_sizes, activation=mdl_opts.activation,
                           solver=mdl_opts.solver, alpha=mdl_opts.alpha)
    mdl, time_to_train = train_and_time(mdl, x_train, y_train)
    y_pred = mdl.predict(x_test)

    return mdl, y_pred, time_to_train


########################################################################################################################
# evaluate methods
########################################################################################################################

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot test and pred
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def make_plots_pred_and_actual(y_test, y_pred, model_description, model_id):
    tuple_temp = tuple(zip(y_test, y_pred))
    tuple_sorted = sorted(tuple_temp, key=lambda test: test[0])
    test_vals = [sample[0] for sample in tuple_sorted]
    pred_vals = [sample[1] for sample in tuple_sorted]
    sample_ids = range(len(y_test))

    # ======================================================================================================================
    # plot 1: predictions and actual
    # ======================================================================================================================

    plt.figure(model_id)
    plt.scatter(sample_ids, test_vals)
    plt.scatter(sample_ids, pred_vals)
    plt.title(model_description)
    plt.xlabel("Sorted Sample ID")
    plt.ylabel("Fuel Economy")
    plt.legend(["Actual", "Prediction"])
    plt.show()

    # ======================================================================================================================
    # plot 2: predictions v actual
    # ======================================================================================================================

    xmin = test_vals[0]
    xmax = test_vals[-1]
    xrange=np.linspace(xmin,xmax,2)

    plt.figure(100 + model_id)
    plt.plot(xrange, xrange, color='black')
    plt.scatter(test_vals, pred_vals)
    plt.title(model_description)
    plt.xlabel("Actual Fuel Economy")
    plt.ylabel("Predicted Fuel Economy")
    plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# compute/display errors
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def compute_model_performance(y_test, y_pred, time_to_train, model_description):
    print('-----------------------------------------------------------------------------------------------------')
    print('Summary of Model (' + model_description + ')')
    print('-----------------------------------------------------------------------------------------------------\n')

    print('Time to train (hh:mm:ss): ' + time_to_train.hrs + ":" + time_to_train.mins + ":" + time_to_train.secs + '\n')

    max_error = max(abs(y_pred - y_test))
    print('Max Error:', max_error)
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', mse)
    mape = mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error:', mape)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# helping function: convert time
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class TimeStr:
    def __init__(self):
        self.hrs = None
        self.mins = None
        self.secs = None


def convert_sec_to_hr_min_sec(secs_raw):
    temp = secs_raw * 1 / 60 * 1 / 60
    hrs = math.floor(temp)
    temp = (temp - hrs) * 60
    mins = math.floor(temp)
    temp = (temp - mins) * 60
    secs = math.floor(temp)

    time = TimeStr()

    if hrs < 10:
        time.hrs = '0' + str(hrs)
    else:
        time.hrs = str(hrs)

    if mins < 10:
        time.mins = '0' + str(mins)
    else:
        time.mins = str(mins)

    if secs < 10:
        time.secs = '0' + str(secs)
    else:
        time.secs = str(secs)

    return time


########################################################################################################################
# main algorithm
########################################################################################################################

def main():
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inputs
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    name_of_datafile = 'carData.txt'
    name_of_response_term = 'FuelEcon'

    # ----------------------------------------------------------------------------------------------------------------------
    # training models
    # ----------------------------------------------------------------------------------------------------------------------

    # IDs:
    # 101: Optimizer for Least Squares Regression Model
    # 102: Optimizer for Regression Tree Model
    # 103: Optimizer for Support Vector Regression Model
    # 104: Optimizer for Gaussian Process Regressor Model
    # 105: Optimizer for Multi-Layer Perceptron (NN) Regressor Model
    # 1: Default Least Squares Linear Regression Model
    # 2: Customized Least Squares Linear Regression Model
    # 3: Customized Least Squares Polynomial Regression Model
    # 4: Default Regression Tree
    # 5: Customized Regression Tree
    # 6: Pruned Customized Regression Tree ?????
    # 7: Default Support Vector Regression Model
    # 8: Customized Linear Support Vector Regression Model
    # 9: Customized Polynomial Support Vector Regression Model
    # 10: Customized Radial Basis Function Support Vector Regression Model
    # 11: Default Gaussian Process Regression Model
    # 12: Customized Constant Gaussian Process Regression Model
    # 13: Customized Radial Basis Function Gaussian Process Regression Model
    # 14: Customized White Noise Gaussian Process Regression Model
    # 15: Default Multi-Layer Perceptron (NN) Regressor Model
    # 16: Customized 1-layer Multi-Layer Perceptron (NN) Regressor Model
    # 17: Customized 2-layer Multi-Layer Perceptron (NN) Regressor Model
    # 18: Customized 3-layer Multi-Layer Perceptron (NN) Regressor Model
    # 19: Customized 4-layer Multi-Layer Perceptron (NN) Regressor Model
    list_models_to_run = [1, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # read in data
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    x_train, y_train, x_test, y_test = get_train_and_test_one_hot(name_of_datafile, name_of_response_term)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # run algorithms
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # optimization algorithms
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ======================================================================================================================
    # optimize least squares linear regression model
    # ======================================================================================================================

    model_id = 101
    model_description = 'Optimize Least Squares Linear Regression Model'

    if model_id in list_models_to_run:
        introduce_model(model_description)
        optimizer_least_squares(x_train, y_train, x_test, y_test)

    # ======================================================================================================================
    # optimize support vector regression model
    # ======================================================================================================================

    model_id = 102
    model_description = 'Optimize Regression Tree Model'

    if model_id in list_models_to_run:
        introduce_model(model_description)
        optimizer_decision_tree(x_train, y_train, x_test, y_test)

    # ======================================================================================================================
    # optimize support vector regression model
    # ======================================================================================================================

    model_id = 103
    model_description = 'Optimize Support Vector Regression Model'

    if model_id in list_models_to_run:
        introduce_model(model_description)
        optimizer_support_vector_regression(x_train, y_train, x_test, y_test)

    # ======================================================================================================================
    # optimize gaussian process regressor model
    # ======================================================================================================================

    model_id = 104
    model_description = 'Optimize Gaussian Process Regressor Model'

    if model_id in list_models_to_run:
        introduce_model(model_description)
        optimizer_gaussian_process_regressor(x_train, y_train, x_test, y_test)

    # ======================================================================================================================
    # optimize multi-layer perceptron (NN) regressor model
    # ======================================================================================================================

    model_id = 105
    model_description = 'Optimize Multi-Layer Perceptron (NN) Regressor Model'

    if model_id in list_models_to_run:
        introduce_model(model_description)
        optimizer_neural_network(x_train, y_train, x_test, y_test)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # linear regression models
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ======================================================================================================================
    # default least squares linear regression model
    # ======================================================================================================================

    model_id = 1
    model_description = 'Default Least Squares Linear Regression Model'
    mdl_opts = ModelOptions()

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_least_squares(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 5.899576157662736
    Mean Squared Error: 1.714599509631553
    Mean Absolute Error: 0.8864971639055622
    '''

    # ======================================================================================================================
    # customized least squares linear regression model
    # ======================================================================================================================

    model_id = 2
    model_description = 'Customized Least Squares Linear Regression Model'
    mdl_opts = ModelOptions()
    mdl_opts.scale = 'minmax'

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_least_squares(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Max Error: 12.60414863
    Mean Squared Error: 1.7145995096315494
    Mean Absolute Error: 0.8864971639055628
    '''

    # ======================================================================================================================
    # customized least squares polynomial regression model
    # ======================================================================================================================

    model_id = 3
    model_description = 'Customized Least Squares Polynomial Regression Model'
    mdl_opts = ModelOptions()
    mdl_opts.scale = 'none'
    mdl_opts.degree = 3

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_least_squares(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 73.43354671099598
    Mean Squared Error: 63.49463046085521
    Mean Absolute Error: 3.4930095416658236
    '''

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # regression tree models
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ======================================================================================================================
    # default regression tree model
    # ======================================================================================================================

    model_id = 4
    model_description = 'Default Regression Tree Model'
    mdl_opts = ModelOptions()

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_decision_tree_regressor(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 6.253100000000002
    Mean Squared Error: 1.2218416664459877
    Mean Absolute Error: 0.6992217592592592
    '''

    # ======================================================================================================================
    # customized regression tree model
    # ======================================================================================================================

    model_id = 5
    model_description = 'Customized Regression Tree Model'
    mdl_opts = ModelOptions()
    mdl_opts.criterion = 'poisson'
    mdl_opts.ccp_alpha = 0.00016418899255942252
    mdl_opts.max_depth = 8
    mdl_opts.min_samples_leaf = 2
    mdl_opts.min_samples_split = 4

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_decision_tree_regressor(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 4.0174
    Mean Squared Error: 0.9954437996882717
    Mean Absolute Error: 0.6656357407407408
    '''

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # support vector regression models
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ======================================================================================================================
    # default SVR model
    # ======================================================================================================================

    model_id = 7
    model_description = 'Default Support Vector Regression Model'
    mdl_opts = ModelOptions()
    mdl_opts.scale = 'standardized'

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_support_vector_regression(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 12.771721400904976
    Mean Squared Error: 1.9518844641245077
    Mean Absolute Error: 0.7995781497859323
    '''

    # ======================================================================================================================
    # customized linear SVR model
    # ======================================================================================================================

    model_id = 8
    model_description = 'Customized Linear Support Vector Regression Model'
    mdl_opts = ModelOptions()
    mdl_opts.scale = 'standardized'
    mdl_opts.kernel = 'linear'
    mdl_opts.C = 0.1
    mdl_opts.coef0 = 1e-07
    mdl_opts.epsilon = 0.03162277660168379

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_support_vector_regression(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:03
    Max Error: 6.908129761843792
    Mean Squared Error: 2.0705261471815484
    Mean Absolute Error: 0.9026461731171309
    '''

    # ======================================================================================================================
    # customized polynomial SVR model
    # ======================================================================================================================

    model_id = 9
    model_description = 'Customized Polynomial Support Vector Regression Model'
    mdl_opts = ModelOptions()
    mdl_opts.scale = 'standardized'
    mdl_opts.kernel = 'poly'
    mdl_opts.C = 0.5
    mdl_opts.coef0 = 1.0
    mdl_opts.epsilon = 1e-1
    mdl_opts.degree = 4

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_support_vector_regression(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 12.795771510276992
    Mean Squared Error: 0.7039639446611283
    Mean Absolute Error: 0.5911225311545939
    '''

    # ======================================================================================================================
    # customized rbf SVR model
    # ======================================================================================================================

    model_id = 10
    model_description = 'Customized Radial Basis Function Support Vector Regression Model'
    mdl_opts = ModelOptions()
    mdl_opts.scale = 'standardized'
    mdl_opts.kernel = 'rbf'
    mdl_opts.C = 8.6
    mdl_opts.coef0 = 1.0
    mdl_opts.epsilon = 1e-3

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_support_vector_regression(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 9.021232370086146
    Mean Squared Error: 7.871019861959437
    Mean Absolute Error: 2.3378450528629346
    '''

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # gaussian process regression models
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ======================================================================================================================
    # default gaussian process regression model
    # ======================================================================================================================

    model_id = 11
    model_description = 'Default Gaussian Process Regression Model'
    mdl_opts = ModelOptions()
    mdl_opts.scale = 'minmax'

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_gaussian_process_regression(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 5.576889508703347
    Mean Squared Error: 1.1355749122992131
    Mean Absolute Error: 0.7165751816209149
    '''

    # ======================================================================================================================
    # Customized constant gaussian process regression model
    # ======================================================================================================================

    model_id = 12
    model_description = 'Customized Constant Gaussian Process Regression Model'
    mdl_opts = ModelOptions()
    mdl_opts.scale = 'minmax'
    # mdl_opts.kernel='constant','rbf','whitnoise'
    mdl_opts.kernel = 'constant'
    mdl_opts.gp_kernel_const = 10
    mdl_opts.normalize_y = False

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_gaussian_process_regression(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 4.118053339958188
    Mean Squared Error: 0.9151400462468312
    Mean Absolute Error: 0.6594975202401476
    '''

    # ======================================================================================================================
    # Customized rbf gaussian process regression model
    # ======================================================================================================================

    model_id = 13
    model_description = 'Customized Radial Basis Function Gaussian Process Regression Model'
    mdl_opts = ModelOptions()
    mdl_opts.scale = 'minmax'
    # mdl_opts.kernel='constant','rbf','whitnoise'
    mdl_opts.kernel = 'rbf'
    mdl_opts.gp_kernel_const = 1e3
    mdl_opts.length_scale = 1
    mdl_opts.normalize_y = True

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_gaussian_process_regression(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 4.363174695919536
    Mean Squared Error: 0.9311179504755519
    Mean Absolute Error: 0.6608275224130139
    '''

    # ======================================================================================================================
    # Customized white noise gaussian process regression model
    # ======================================================================================================================

    model_id = 14
    model_description = 'Customized White Noise Gaussian Process Regression Model'
    mdl_opts = ModelOptions()
    mdl_opts.scale = 'minmax'
    mdl_opts.kernel = 'whitenoise'
    mdl_opts.noise_level = 1
    mdl_opts.noise_level_bounds_min = 1e-3
    mdl_opts.noise_level_bounds_max = 1e3
    mdl_opts.normalize_y = True

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_gaussian_process_regression(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:08
    Max Error: 5.882560527053197
    Mean Squared Error: 1.7137700519758994
    Mean Absolute Error: 0.8829901578196386
    '''

    # ======================================================================================================================
    # default neural network regression model
    # ======================================================================================================================

    model_id = 15
    model_description = 'Default Multi-Layer Perceptron (NN) Regressor Model'
    mdl_opts = ModelOptions()

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_neural_network(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 7.361025682527286
    Mean Squared Error: 5.060671193476527
    Mean Absolute Error: 1.8830979751122048
    '''
    
    # ======================================================================================================================
    # Customized 1-layer neural network regression model
    # ======================================================================================================================

    model_id = 16
    model_description = 'Customized 1-layer Multi-Layer Perceptron (NN) Regressor Model'
    mdl_opts = ModelOptions()
    mdl_opts.hidden_layer_sizes = (50,)
    mdl_opts.activation = 'identity'
    mdl_opts.solver = 'lbfgs'
    mdl_opts.alpha = 0.1

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_neural_network(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 6.505383418995123
    Mean Squared Error: 1.8912190488474396
    Mean Absolute Error: 0.9375072249066733
    '''
    
    # ======================================================================================================================
    # Customized 2-layer neural network regression model
    # ======================================================================================================================

    model_id = 17
    model_description = 'Customized 2-layer Multi-Layer Perceptron (NN) Regressor Model'
    mdl_opts = ModelOptions()
    mdl_opts.hidden_layer_sizes = (100, 1000)
    mdl_opts.activation = 'tanh'
    mdl_opts.solver = 'adam'
    mdl_opts.alpha = 0.1

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_neural_network(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 9.552199522631673
    Mean Squared Error: 9.270577015748268
    Mean Absolute Error: 2.510595663925085
    '''
    
    # ======================================================================================================================
    # Customized 3-layer neural network regression model
    # ======================================================================================================================

    model_id = 18
    model_description = 'Customized 3-layer Multi-Layer Perceptron (NN) Regressor Model'
    mdl_opts = ModelOptions()
    mdl_opts.hidden_layer_sizes = (1000, 500, 1000)
    mdl_opts.activation = 'logistic'
    mdl_opts.solver = 'adam'
    mdl_opts.alpha = 0.01

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_neural_network(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:06
    Max Error: 6.50084837648291
    Mean Squared Error: 4.789771957487149
    Mean Absolute Error: 1.7740611687769703
    '''
    
    # ======================================================================================================================
    # Customized 4-layer neural network regression model
    # ======================================================================================================================

    model_id = 19
    model_description = 'Customized 4-layer Multi-Layer Perceptron (NN) Regressor Model'
    mdl_opts = ModelOptions()
    mdl_opts.hidden_layer_sizes = (500, 50, 50, 100)
    mdl_opts.activation = 'tanh'
    mdl_opts.solver = 'adam'
    mdl_opts.alpha = 0.1

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        introduce_model(model_description)
        mdl, y_pred, time_to_train = get_mdl_neural_network(x_train, y_train, x_test, mdl_opts)

        # ----------------------------------------------------------------------------------------------------------------------
        # output performance
        # ----------------------------------------------------------------------------------------------------------------------

        make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
        compute_model_performance(y_test, y_pred, time_to_train, model_description)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 8.839366969927596
    Mean Squared Error: 6.935221892291282
    Mean Absolute Error: 2.109328633468339
    '''

    breakpoint_here_for_debug = True


if __name__ == "__main__":
    main()
