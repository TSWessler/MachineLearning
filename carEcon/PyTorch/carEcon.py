'''
Name:
    carEcon

Version:
    wessler
    2025 January 7
    1st version

Description:
    *uses PyTorch Regression NN's to predict fuel economy of cars given several features
    *reads the Table in carEcon.txt
    *carEcon data adapted from MATLAB ML course


Used by:
    *NOTHING--this is the code to run

Uses:
    *NOTHING

To run:
    *at the beginning of main(), fill in "list_models_to_run" with a list of IDs for which model to run
    *if id=101 is chosen, then this is the optimization mode--go into the "optimize_mdl_custom_nn" to enter the all the model options to be tried

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
for optimization, probably should re-shuffle train/test data and run replications of each method
'''

import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tqdm
import copy
import itertools


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
########################################################################################################################


########################################################################################################################
# read/prepare data
########################################################################################################################

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

    # convert to 2d PyTorch tensors
    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    x_test = torch.tensor(x_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    return x_train, y_train, x_test, y_test


########################################################################################################################
# setting up model
########################################################################################################################

# ======================================================================================================================
# class for making options for model
# ======================================================================================================================
class ModelOptions:
    def __init__(self,model_id,model_description,num_features):
        self.model_id = model_id
        self.model_description = model_description
        self.num_epochs = None
        self.batch_size = None
        self.hide_epoch_progress = None
        self.loss_fn_id = None
        self.optimizer_id = None
        self.learning_rate = None
        self.num_layers = None
        self.list_meths = None
        self.list_act_funcs = None
        self.list_layer_sizes = None
        self.num_features = num_features
        self.loss_fn = None
        self.optimizer = None


# ======================================================================================================================
# loss and optimization functions
# ======================================================================================================================
def get_loss_func(loss_fn_id):

    if loss_fn_id == 1: #Mean Square Error Loss
        return nn.MSELoss()
    elif loss_fn_id == 2: #L1 (Mean Absolute Error) Loss
        return nn.L1Loss()
    elif loss_fn_id == 3: #Huber Loss
        return nn.HuberLoss(reduction='mean', delta=1.0)


def get_optimize_meth(optimizer_id ,parameters, learning_rate):

    if optimizer_id == 1:
        return optim.Adam(parameters, lr=learning_rate)
    elif optimizer_id == 2:
        return optim.SGD(parameters, lr=learning_rate)


########################################################################################################################
# running model
########################################################################################################################

# ======================================================================================================================
# run model id
# ======================================================================================================================

def run_model_id(x_train,y_train,x_test,y_test,mdl_opts):

    #unpack
    model_id = mdl_opts.model_id
    model_description = mdl_opts.model_description
    
    # ----------------------------------------------------------------------------------------------------------------------
    # train, predict
    # ----------------------------------------------------------------------------------------------------------------------

    print('\n\n====================================================================================================')
    print(f"Model {model_id}: {model_description}")
    print('====================================================================================================\n')
    print('training...\n')
    mdl, y_pred, time_to_train = get_mdl_custom_nn(x_train, y_train, x_test, y_test, mdl_opts)

    # ----------------------------------------------------------------------------------------------------------------------
    # output performance
    # ----------------------------------------------------------------------------------------------------------------------

    make_plots_pred_and_actual(y_test, y_pred, model_description, model_id)
    print('-----------------------------------------------------------------------------------------------------')
    print(f"Summary of Model {model_id} ({model_description})")
    print('-----------------------------------------------------------------------------------------------------\n')
    compute_model_performance(y_test, y_pred, time_to_train)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# make and run training on models in optimization mode
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def optimize_mdl_custom_nn(x_train, y_train, x_test, y_test, mdl_opts):

    list_loss_fn_ids = [1,2,3]
    #list_loss_fn_ids = [1]
    list_optimizer_ids = [1,2]
    #list_optimizer_ids = [1]
    list_learning_rates = [1e-2,1e-3,1e-4]
    #list_learning_rates = [1e-2]
    list_activation_funcs = ["Sigmoid", "ReLU", "Tanh", "Softmax"]
    #list_activation_funcs = ["ReLU"]
    list_forw_funcs = ["Linear"]
    list_num_layers = [3]
    list_layer_sizes_to_try = [1e1,5e1,1e2,5e2,1e3]
    #list_layer_sizes_to_try = [10,20,30,40,50,60,70,80,90,100]
    code_firm_layers_or_sweep = 1 #0=sweep, 1=firm list
    firm_layers_list = [24,12,6]

    mse_optimum = np.inf
    mdl_optimum = None
    loss_fn_id_optimum = None
    optimizer_id_optimum = None
    learning_rate_id_optimum = None
    activation_func_optimum = None
    meth_optimum = None
    run_id = 0
    run_id_optimum = None

    for loss_fn_id in list_loss_fn_ids:
        mdl_opts.loss_fn_id = loss_fn_id
        for optimizer_id in list_optimizer_ids:
            mdl_opts.optimizer_id = optimizer_id
            for learning_rate in list_learning_rates:
                mdl_opts.learning_rate = learning_rate
                for activation_func in list_activation_funcs:
                    for meth in list_forw_funcs:
                        for num_layers in list_num_layers:
                            mdl_opts.num_layers = num_layers

                            list_meths=[]
                            list_act_funcs = []
                            for ii in range(num_layers+1):
                                list_meths.append(meth)
                                if ii < num_layers:
                                    list_act_funcs.append(activation_func)
                            mdl_opts.list_meths = list_meths
                            mdl_opts.list_act_funcs = list_act_funcs

                            if code_firm_layers_or_sweep == 0:
                                list_combos = list(itertools.product(list_layer_sizes_to_try, repeat=num_layers))
                            else:
                                list_combos = [tuple(firm_layers_list)]
                            for combo in list_combos:
                                run_id += 1
                                mdl_opts.list_layer_sizes = list(combo)
                                mdl_opts.list_layer_sizes = [int(x) for x in mdl_opts.list_layer_sizes]

                                # make model layers
                                layers = []
                                for layer in range(mdl_opts.num_layers + 1):
                                    if layer == 0:
                                        layer_in = mdl_opts.num_features
                                    else:
                                        layer_in = mdl_opts.list_layer_sizes[layer - 1]
                                    if layer == mdl_opts.num_layers:
                                        layer_out = 1
                                    else:
                                        layer_out = mdl_opts.list_layer_sizes[layer]
                                    if mdl_opts.list_meths[layer] == "Linear":
                                        layers.append(nn.Linear(layer_in, layer_out))
                                    if layer < mdl_opts.num_layers:
                                        if mdl_opts.list_act_funcs[layer] == "Sigmoid":
                                            layers.append(nn.Sigmoid())
                                        elif mdl_opts.list_act_funcs[layer] == "ReLU":
                                            layers.append(nn.ReLU())
                                        elif mdl_opts.list_act_funcs[layer] == "Tanh":
                                            layers.append(nn.Tanh())
                                        elif mdl_opts.list_act_funcs[layer] == "Softmax":
                                            layers.append(nn.Softmax(dim=-1))

                                # make model
                                mdl = nn.Sequential(*layers)

                                # get loss func and optimization method
                                mdl_opts.loss_fn = get_loss_func(mdl_opts.loss_fn_id)
                                mdl_opts.optimizer = get_optimize_meth(mdl_opts.optimizer_id, mdl.parameters(), mdl_opts.learning_rate)

                                # =========================  TRAIN   =========================
                                print("\n\n")
                                print('-------------------------------------------------------------------------------')
                                print(f"Model ID {run_id} description:")
                                print(f"loss function id: {mdl_opts.loss_fn_id}")
                                print(f"optimizer id: {mdl_opts.loss_fn_id}")
                                print(f"learning rate: {mdl_opts.learning_rate}")
                                print(f"activation function: {activation_func}")
                                print(f"layer structure: {mdl_opts.list_layer_sizes}")
                                print('-------------------------------------------------------------------------------')
                                print('\ntraining...\n')
                                mdl, time_to_train = train_and_time(mdl, x_train, y_train, x_test, y_test, mdl_opts)

                                mdl.eval()
                                with torch.inference_mode():
                                    y_pred = mdl(x_test)
                                    y_pred = y_pred.squeeze()

                                try:
                                    mse = mean_squared_error(y_test, y_pred)
                                except:
                                    mse = np.inf
                                if mse < mse_optimum:
                                    mse_optimum = mse
                                    mdl_optimum = mdl
                                    loss_fn_id_optimum = loss_fn_id
                                    optimizer_id_optimum = optimizer_id
                                    learning_rate_id_optimum = learning_rate
                                    activation_func_optimum = activation_func
                                    meth_optimum = meth
                                    run_id_optimum = run_id

                                print('===============================================================================')
                                print("Model Performance:")
                                print('===============================================================================')
                                compute_model_performance(y_test, y_pred, time_to_train)

                                print(f"\nMSE (best to date): {mse_optimum} (ID: {run_id_optimum})")
                                print(f"MSE (this param set): {mse}")

    print('\n\n')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f"Summary of Optimum Model")
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f"MSE: {mse_optimum}\nID: {run_id_optimum}")
    print(f"loss_fn_id: {loss_fn_id_optimum}\noptimizer_id: {optimizer_id_optimum}")
    print(f"learning_rate: {learning_rate_id_optimum}")
    print(f"activation_func: {activation_func_optimum}\nforw_meth: {meth_optimum}")
    print("model:")
    print(mdl_optimum)

    return mdl_optimum


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# make model and run model trainer in regular mode
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_mdl_custom_nn(x_train, y_train, x_test, y_test, mdl_opts):

    # make model layers
    layers = []
    for layer in range(mdl_opts.num_layers+1):
        if layer == 0:
            layer_in = mdl_opts.num_features
        else:
           layer_in = mdl_opts.list_layer_sizes[layer-1]
        if layer == mdl_opts.num_layers:
            layer_out = 1
        else:
            layer_out = mdl_opts.list_layer_sizes[layer]
        if mdl_opts.list_meths[layer] == "Linear":
            layers.append(nn.Linear(layer_in, layer_out))
        if layer < mdl_opts.num_layers:
                if mdl_opts.list_act_funcs[layer] == "Sigmoid":
                    layers.append(nn.Sigmoid())
                elif mdl_opts.list_act_funcs[layer] == "ReLU":
                    layers.append(nn.ReLU())
                elif mdl_opts.list_act_funcs[layer] == "Tanh":
                    layers.append(nn.Tanh())
                elif mdl_opts.list_act_funcs[layer] == "Softmax":
                    layers.append(nn.Softmax(dim=-1))

    # make model
    mdl = nn.Sequential(*layers)

    # get loss func and optimization method
    mdl_opts.loss_fn = get_loss_func(mdl_opts.loss_fn_id)
    mdl_opts.optimizer = get_optimize_meth(mdl_opts.optimizer_id, mdl.parameters(), mdl_opts.learning_rate)

    # =========================  TRAIN   =========================
    mdl, time_to_train = train_and_time(mdl, x_train, y_train, x_test, y_test, mdl_opts)

    mdl.eval()
    with torch.inference_mode():
        y_pred = mdl(x_test)
        y_pred = y_pred.squeeze()

    return mdl, y_pred, time_to_train


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# train model (and time training)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def train_and_time(mdl, x_train, y_train, x_test, y_test, mdl_opts):

    # unpack mdl_opts
    loss_fn = mdl_opts.loss_fn
    optimizer = mdl_opts.optimizer
    num_epochs = mdl_opts.num_epochs
    batch_size = mdl_opts.batch_size
    hide_epoch_progress = mdl_opts.hide_epoch_progress

    # initialize 1st batch
    batch_start = torch.arange(0, len(x_train), batch_size)

    # initialize for finding optimum model
    loss_optimum = np.inf
    weights_optimum = None
    
    # initialize loss vals for plot
    stride_plot = math.ceil(num_epochs/10)
    epoch_vals = []
    loss_vals_train = []
    loss_vals_test = []

    # ----- START TIMER -----
    time_start = time.time()

    # training loop
    for epoch in range(num_epochs):
        mdl.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=hide_epoch_progress) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                x_batch = x_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]
                # forward pass
                y_pred = mdl(x_batch)
                y_pred = y_pred.squeeze()
                loss_train = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss_train.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(loss=float(loss_train))

        # evaluate accuracy at end of each epoch
        mdl.eval()

        # forward pass test data
        with torch.inference_mode():
            y_pred = mdl(x_test)
            y_pred = y_pred.squeeze()
        # loss on test data
        loss_test = loss_fn(y_pred, y_test)
        loss_test = float(loss_test)
        if loss_test < loss_optimum:
            loss_optimum = loss_test
            weights_optimum = copy.deepcopy(mdl.state_dict())
        
        if epoch % stride_plot == 0:
            epoch_vals.append(epoch)
            loss_vals_train.append(loss_train.detach().numpy())
            loss_vals_test.append(loss_test)

    # set model to optimum
    try:
        mdl.load_state_dict(weights_optimum)
    except:
        print("\n\n\nThis model is giving nan's!!!!!\n\n\n")

    # ----- END TIMER -----
    time_end = time.time()
    elapse_time = time_end - time_start
    time_to_train = convert_sec_to_hr_min_sec(elapse_time)

    make_plot_loss_curves(epoch_vals, loss_vals_train, loss_vals_test, mdl_opts.model_id)

    return mdl, time_to_train

# ======================================================================================================================
# plot loss curves
# ======================================================================================================================

def make_plot_loss_curves(epoch_vals, loss_vals_train, loss_vals_test, model_id):
    plt.figure(1000+model_id)
    plt.plot(epoch_vals, loss_vals_train, label="Train loss")
    plt.plot(epoch_vals, loss_vals_test, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


########################################################################################################################
# evaluate models
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

def compute_model_performance(y_test, y_pred, time_to_train):
    
    print(f"Time to train (hh:mm:ss): {time_to_train.hrs}:{time_to_train.mins}:{time_to_train.secs}\n")

    max_error = max(abs(y_pred - y_test)).item()
    print('Max Error:', max_error)
    try:
        mse = mean_squared_error(y_test, y_pred)
    except:
        mse = np.inf
    print('Mean Squared Error:', mse)
    try:
        mape = mean_absolute_error(y_test, y_pred)
    except:
        mape = np.inf
    print('Mean Absolute Error:', mape)


# ======================================================================================================================
# helping function: convert time
# ======================================================================================================================

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
    # 0: 0-layer NN
    # 1: 1-layer NN (600-node)
    # 2: 2-layer NN (10-100)
    # 3: 3-layer NN (50-10-10)
    # 4: 4-layer NN (50-100-50-500)
    # 5: 5-layer NN (24-12-6 Pyramid)
    # 6: 24-12-6-1 Pyramid NN
    # 101: optimize NN
    list_models_to_run = [0,1,2,3,4,5,6]

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # read in data
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    x_train, y_train, x_test, y_test = get_train_and_test_one_hot(name_of_datafile, name_of_response_term)
    num_features = x_test.shape[1]

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # run algorithms
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # NN regression models
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ======================================================================================================================
    # Model 0: 0-Layer Regression
    # ======================================================================================================================

    model_id = 0
    model_description = '0-Layer Regression'
    mdl_opts = ModelOptions(model_id,model_description,num_features)
    mdl_opts.num_epochs = 100
    mdl_opts.batch_size = 10
    mdl_opts.hide_epoch_progress = True
    mdl_opts.loss_fn_id = 1
    mdl_opts.optimizer_id = 1
    mdl_opts.learning_rate = 0.01
    mdl_opts.num_layers = 0
    mdl_opts.list_meths = ["Linear"]

    if model_id in list_models_to_run:
        run_model_id(x_train, y_train, x_test, y_test, mdl_opts)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 5.91651725769043
    Mean Squared Error: 1.6709822
    Mean Absolute Error: 0.9354184
    '''


    # ======================================================================================================================
    # Model 1: 1-Layer Regression (600-node)
    # ======================================================================================================================

    model_id = 1
    model_description = '1-Layer Regression (600-node)'
    mdl_opts = ModelOptions(model_id,model_description,num_features)
    mdl_opts.num_epochs = 100
    mdl_opts.batch_size = 10
    mdl_opts.hide_epoch_progress = True
    mdl_opts.loss_fn_id = 1
    mdl_opts.optimizer_id = 1
    mdl_opts.learning_rate = 0.001
    mdl_opts.num_layers = 1
    mdl_opts.list_meths = ["Linear","Linear"]
    mdl_opts.list_act_funcs = ["ReLU"]
    mdl_opts.list_layer_sizes = [600]

    if model_id in list_models_to_run:
        run_model_id(x_train, y_train, x_test, y_test, mdl_opts)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:01
    Max Error: 5.591792106628418
    Mean Squared Error: 1.6485873
    Mean Absolute Error: 0.9333618
    '''


    # ======================================================================================================================
    # Model 2: 2-Layer Regression (10-100)
    # ======================================================================================================================

    model_id = 2
    model_description = '2-Layer Regression (10-100)'
    mdl_opts = ModelOptions(model_id,model_description,num_features)
    mdl_opts.num_epochs = 100
    mdl_opts.batch_size = 10
    mdl_opts.hide_epoch_progress = True
    mdl_opts.loss_fn_id = 1
    mdl_opts.optimizer_id = 1
    mdl_opts.learning_rate = 0.01
    mdl_opts.num_layers = 2
    mdl_opts.list_meths = ["Linear","Linear","Linear"]
    mdl_opts.list_act_funcs = ["ReLU","ReLU"]
    mdl_opts.list_layer_sizes = [10,100]

    if model_id in list_models_to_run:
        run_model_id(x_train, y_train, x_test, y_test, mdl_opts)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:01
    Max Error: 5.081811904907227
    Mean Squared Error: 1.3437551
    Mean Absolute Error: 0.83356476
    '''


    # ======================================================================================================================
    # Model 3: 3-Layer Regression (50-10-10)
    # ======================================================================================================================

    model_id = 3
    model_description = '3-Layer Regression (50-10-10)'
    mdl_opts = ModelOptions(model_id,model_description,num_features)
    mdl_opts.num_epochs = 100
    mdl_opts.batch_size = 10
    mdl_opts.hide_epoch_progress = True
    mdl_opts.loss_fn_id = 1
    mdl_opts.optimizer_id = 1
    mdl_opts.learning_rate = 0.01
    mdl_opts.num_layers = 3
    mdl_opts.list_meths = ["Linear","Linear","Linear","Linear"]
    mdl_opts.list_act_funcs = ["ReLU","ReLU","ReLU"]
    mdl_opts.list_layer_sizes = [50,10,10]

    if model_id in list_models_to_run:
        run_model_id(x_train, y_train, x_test, y_test, mdl_opts)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:01
    Max Error: 4.085099220275879
    Mean Squared Error: 1.4114199
    Mean Absolute Error: 0.9078722
    '''

    # ======================================================================================================================
    # Model 4: 4-Layer Regression (50-100-50-500)
    # ======================================================================================================================

    model_id = 4
    model_description = '4-Layer Regression'
    mdl_opts = ModelOptions(model_id,model_description,num_features)
    mdl_opts.num_epochs = 100
    mdl_opts.batch_size = 10
    mdl_opts.hide_epoch_progress = True
    mdl_opts.loss_fn_id = 1
    mdl_opts.optimizer_id = 1
    mdl_opts.learning_rate = 0.01
    mdl_opts.num_layers = 4
    mdl_opts.list_meths = ["Linear","Linear","Linear","Linear","Linear"]
    mdl_opts.list_act_funcs = ["ReLU","ReLU","ReLU","ReLU"]
    mdl_opts.list_layer_sizes = [50,100,50,500]

    if model_id in list_models_to_run:
        run_model_id(x_train, y_train, x_test, y_test, mdl_opts)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:02
    Max Error: 4.880510330200195
    Mean Squared Error: 1.1418074
    Mean Absolute Error: 0.7090044
    '''

    # ======================================================================================================================
    # Model 5: 5-Layer Regression (10-500-10-1000-500)
    # ======================================================================================================================

    model_id = 5
    model_description = '5-Layer Regression (10-500-10-1000-500)'
    mdl_opts = ModelOptions(model_id,model_description,num_features)
    mdl_opts.num_epochs = 100
    mdl_opts.batch_size = 10
    mdl_opts.hide_epoch_progress = True
    mdl_opts.loss_fn_id = 1
    mdl_opts.optimizer_id = 1
    mdl_opts.learning_rate = 0.001
    mdl_opts.num_layers = 5
    mdl_opts.list_meths = ["Linear","Linear","Linear","Linear","Linear","Linear"]
    mdl_opts.list_act_funcs = ["ReLU","ReLU","ReLU","ReLU","ReLU"]
    mdl_opts.list_layer_sizes = [10,500,10,1000,500]

    if model_id in list_models_to_run:
        run_model_id(x_train, y_train, x_test, y_test, mdl_opts)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:06
    Max Error: 5.008203506469727
    Mean Squared Error: 1.5429478
    Mean Absolute Error: 0.9229554
    '''


    # ======================================================================================================================
    # Model 6: 24-12-6 Pyramid NN
    # ======================================================================================================================

    model_id = 6
    model_description = '24-12-6 Pyramid NN'
    mdl_opts = ModelOptions(model_id,model_description,num_features)
    mdl_opts.num_epochs = 100
    mdl_opts.batch_size = 10
    mdl_opts.hide_epoch_progress = True
    mdl_opts.loss_fn_id = 1
    mdl_opts.optimizer_id = 1
    mdl_opts.learning_rate = 0.01
    mdl_opts.num_layers = 3
    mdl_opts.list_meths = ["Linear", "Linear", "Linear", "Linear"]
    mdl_opts.list_act_funcs = ["ReLU", "ReLU", "ReLU"]
    mdl_opts.list_layer_sizes = [24, 12, 6]

    if model_id in list_models_to_run:
        run_model_id(x_train, y_train, x_test, y_test, mdl_opts)
        print('\n\n')

    '''
    Time to train (hh:mm:ss): 00:00:00
    Max Error: 5.4578962326049805
    Mean Squared Error: 1.396234
    Mean Absolute Error: 0.7848712
    '''

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # optimization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ======================================================================================================================
    # Model 101: Optimization
    # ======================================================================================================================

    model_id = 101
    model_description = 'Optimization'
    mdl_opts = ModelOptions(model_id,model_description,num_features)
    mdl_opts.num_epochs = 100
    mdl_opts.batch_size = 10
    mdl_opts.hide_epoch_progress = True

    if model_id in list_models_to_run:
        # ----------------------------------------------------------------------------------------------------------------------
        # train, predict
        # ----------------------------------------------------------------------------------------------------------------------

        print('\n\n===================================================================================================')
        print('Running Model Optimization')
        print('===================================================================================================\n\n')
        mdl = optimize_mdl_custom_nn(x_train, y_train, x_test, y_test, mdl_opts)

    
    breakpoint_here_for_debug = True


if __name__ == "__main__":
    main()
