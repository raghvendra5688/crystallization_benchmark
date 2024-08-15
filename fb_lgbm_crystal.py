# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Please install 🤗 Transformers as well as some other libraries. Uncomment the following cell and run it.

# %%
import os
import pickle
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import re

import lightgbm
from sklearn import dummy
from sklearn import linear_model
from sklearn import svm
from sklearn import neural_network
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from scipy.stats import loguniform
import scipy
import argparse
from evaluate import load
from misc import grid_search_cv, supervised_learning_steps, get_CV_results,calculate_classification_metrics, load_model

f1_metric,roc_metric,acc_metric,mcc_metric,prec_metric,rec_metric = load("f1"),load("roc_auc"),load("accuracy"),load("matthews_correlation"),load("precision"),load("recall")

# -
#Get the setting with different X_trains and X_tests
train_options = ["Results/crystallization_Ankh-Large_AVG.csv",
                 "Results/crystallization_Ankh_AVG.csv",
                 "Results/crystallization_ProstT5_AVG.csv",
                 "Results/crystallization_ProtT5-XL_AVG.csv",
                 "Results/crystallization_esm2_t12_35M_AVG.csv",
                 "Results/crystallization_esm2_t30_150M_AVG.csv",
                 "Results/crystallization_esm2_t33_650M_AVG.csv",
                 "Results/crystallization_esm2_t36_3B_AVG.csv",
                 "Results/crystallization_esm2_t6_8M_AVG.csv"]
sp_test_options = ["Results/crystallization_sp_Ankh-Large_AVG.csv",
                "Results/crystallization_sp_Ankh_AVG.csv",
                "Results/crystallization_sp_ProstT5_AVG.csv",
                "Results/crystallization_sp_ProtT5-XL_AVG.csv",
                "Results/crystallization_sp_esm2_t12_35M_AVG.csv",
                "Results/crystallization_sp_esm2_t30_150M_AVG.csv",
                "Results/crystallization_sp_esm2_t33_650M_AVG.csv",
                "Results/crystallization_sp_esm2_t36_3B_AVG.csv",
                "Results/crystallization_sp_esm2_t6_8M_AVG.csv"]
tr_test_options = ["Results/crystallization_tr_Ankh-Large_AVG.csv",
                   "Results/crystallization_tr_Ankh_AVG.csv",
                   "Results/crystallization_tr_ProstT5_AVG.csv",
                   "Results/crystallization_tr_ProtT5-XL_AVG.csv",
                   "Results/crystallization_tr_esm2_t12_35M_AVG.csv",
                   "Results/crystallization_tr_esm2_t30_150M_AVG.csv",
                   "Results/crystallization_tr_esm2_t33_650M_AVG.csv",
                   "Results/crystallization_tr_esm2_t36_3B_AVG.csv",
                   "Results/crystallization_tr_esm2_t6_8M_AVG.csv"]
test_options = ["Results/crystallization_test_Ankh-Large_AVG.csv",
                "Results/crystallization_test_Ankh_AVG.csv",
                "Results/crystallization_test_ProstT5_AVG.csv",
                "Results/crystallization_test_ProtT5-XL_AVG.csv",
                "Results/crystallization_test_esm2_t12_35M_AVG.csv",
                "Results/crystallization_test_esm2_t30_150M_AVG.csv",
                "Results/crystallization_test_esm2_t33_650M_AVG.csv",
                "Results/crystallization_test_esm2_t36_3B_AVG.csv",
                "Results/crystallization_test_esm2_t6_8M_AVG.csv"]
data_type_options = ["Ankh-Large","Ankh","ProstT5","ProtT5-XL","esm2_t12_35M","esm2_t30_150M","esm2_t33_650M","esm2_t36_3B","esm2_t6_8M"]

#Get the dictionary of quality metrics
def full_compute_metrics(predictions,labels):
    #predictions = np.argmax(predictions, axis=1)
    return([f1_metric.compute(predictions=predictions, references=labels),\
            roc_metric.compute(prediction_scores=predictions, references=labels),\
            acc_metric.compute(predictions=predictions, references=labels),\
            mcc_metric.compute(predictions=predictions, references=labels),\
            prec_metric.compute(predictions=predictions, references=labels),\
            rec_metric.compute(predictions=predictions, references=labels)])

# +
#Read the train, test, sp and tr labels
train_labels = pd.read_csv("Data/Crystallization/Train_True_Labels.csv",header=None)
train_labels = train_labels.iloc[:,0].astype(int).tolist()
test_labels = pd.read_csv("Data/Crystallization/y_test.csv",header=None)
test_labels = test_labels.iloc[:,0].astype(int).tolist()
sp_test_labels = pd.read_csv("Data/Crystallization/SP_True_Label.csv",header=None)
sp_test_labels = sp_test_labels.iloc[:,0].astype(int).tolist()
tr_test_labels = pd.read_csv("Data/Crystallization/TR_True_Label.csv",header=None)
tr_test_labels = tr_test_labels.iloc[:,0].astype(int).tolist()
#print(train_labels)
#print(test_labels)
#print(sp_test_labels)
#print(tr_test_labels)


#Perform the training for each PLM and test on different test sets
train_output, test_output, sp_test_output, tr_test_output=[],[],[],[]
train_output_predictions, test_output_predictions, sp_test_output_predictions, tr_test_output_predictions=[],[],[],[]
train_out_prediction_probs, test_out_prediction_probs, sp_test_out_prediction_probs, tr_test_out_prediction_probs = [],[],[],[]
for i in range(0,len(train_options)):

    print("Running the classification script for "+data_type_options[i])

    big_df = pd.read_csv(train_options[i],header="infer")
    big_df.columns = "F"+big_df.columns
    dim = big_df.shape[1]

    big_test_df = pd.read_csv(test_options[i],header="infer")
    big_test_df.columns = "F"+big_test_df.columns

    big_sp_test_df = pd.read_csv(sp_test_options[i],header="infer")
    big_sp_test_df.columns = "F"+big_sp_test_df.columns

    big_tr_test_df = pd.read_csv(tr_test_options[i],header="infer")
    big_tr_test_df.columns = "F"+big_tr_test_df.columns

    #Select number of columns 
    X_train = big_df.iloc[:,0:dim-1]
    X_test = big_test_df.iloc[:,0:dim-1]
    X_sp_test = big_sp_test_df.iloc[:,0:dim-1]
    X_tr_test = big_tr_test_df.iloc[:,0:dim-1]

    print("Shape of training set")
    print(X_train.shape)

    # +
    #Build the LightGBM Regression model
    lgbm_model = lightgbm.LGBMClassifier(boosting_type='gbdt',random_state=14, n_jobs=-1, objective="binary", class_weight="balanced")


    # Grid parameters
    params_lgbm = {
                    "n_estimators": scipy.stats.randint(20, 200),
                    "max_depth": scipy.stats.randint(3, 7),
                    "num_leaves": [16, 32, 64],
                    "min_child_samples": scipy.stats.randint(5, 10),
                    "learning_rate": loguniform(1e-4, 1e-1),
                    "subsample": loguniform(0.8, 1e0),
                    "colsample_bytree": [0.01, 0.05, 0.1],
                    "reg_alpha": loguniform(1e-1, 1e1),
                    "reg_lambda": loguniform(1, 1e1)
                   }

    #It will select 100 random combinations for the CV and do 5-fold CV for each combination
    n_iter=100
    lgbm_gs=supervised_learning_steps("lgbm","f1",data_type_options[i],True,lgbm_model,params_lgbm,\
                                                  X_train,train_labels,n_iter=n_iter,n_splits=5)

    lgbm_gs = load_model("lgbm_models/lgbm_"+data_type_options[i]+"_classifier_gs.pk")
    lgbm_best = lgbm_gs.best_estimator_
    print(lgbm_best)
    
    #Get the predictions for trianing set
    print("Results on Train set for ",data_type_options[i])
    train_predictions = lgbm_best.predict(X_train)
    train_prediction_probs = lgbm_best.predict_proba(X_train)
    out_train = full_compute_metrics(train_predictions, train_labels)
    out_train_dict = dict(map(dict.popitem,out_train))
    out_train_dict["Method"]=data_type_options[i]
    print(out_train_dict)


    #Get the predictions for test set
    print("Results on test set")
    test_predictions = lgbm_best.predict(X_test)
    test_prediction_probs = lgbm_best.predict_proba(X_test)
    out_test = full_compute_metrics(test_predictions, test_labels)
    out_test_dict = dict(map(dict.popitem,out_test))
    out_test_dict["Method"]=data_type_options[i]
    print(out_test_dict)


    #Get the predictions for SP test set
    print("Results on SP test set")
    sp_test_predictions = lgbm_best.predict(X_sp_test)
    sp_test_prediction_probs = lgbm_best.predict_proba(X_sp_test)
    out_sp_test = full_compute_metrics(sp_test_predictions, sp_test_labels)
    out_sp_test_dict = dict(map(dict.popitem,out_sp_test))
    out_sp_test_dict["Method"]=data_type_options[i]
    print(out_sp_test_dict)


    #Get the predictions for TR test set
    print("Results on TR test set")
    tr_test_predictions = lgbm_best.predict(X_tr_test)
    tr_test_prediction_probs = lgbm_best.predict_proba(X_tr_test)
    out_tr_test = full_compute_metrics(tr_test_predictions, tr_test_labels)
    out_tr_test_dict = dict(map(dict.popitem,out_tr_test))
    out_tr_test_dict["Method"]=data_type_options[i]
    print(out_tr_test_dict)

    train_output.append(out_train_dict)
    test_output.append(out_test_dict)
    sp_test_output.append(out_sp_test_dict)
    tr_test_output.append(out_tr_test_dict)

    #Write the predictions 
    train_output_predictions.append(train_predictions)
    test_output_predictions.append(test_predictions)
    sp_test_output_predictions.append(sp_test_predictions)
    tr_test_output_predictions.append(tr_test_predictions)

    #Write the prediction probabilities
    train_out_prediction_probs.append(train_prediction_probs[:,1].tolist())
    test_out_prediction_probs.append(test_prediction_probs[:,1].tolist())
    sp_test_out_prediction_probs.append(sp_test_prediction_probs[:,1].tolist())
    tr_test_out_prediction_probs.append(tr_test_prediction_probs[:,1].tolist())

    del lgbm_gs

# %%
import pandas as pd
from pathlib import Path

train_output_df = pd.DataFrame(train_output)
test_output_df = pd.DataFrame(test_output)
sp_test_output_df = pd.DataFrame(sp_test_output)
tr_test_output_df = pd.DataFrame(tr_test_output)
train_output_pred_df = pd.DataFrame(train_output_predictions).transpose()
train_output_pred_df.columns = data_type_options
test_output_pred_df = pd.DataFrame(test_output_predictions).transpose()
test_output_pred_df.columns = data_type_options
sp_test_output_pred_df = pd.DataFrame(sp_test_output_predictions).transpose()
sp_test_output_pred_df.columns = data_type_options
tr_test_output_pred_df = pd.DataFrame(tr_test_output_predictions).transpose()
tr_test_output_pred_df.columns = data_type_options

#Make the dataset ready for prediction probabilities
train_out_pred_prob_df = pd.DataFrame(train_out_prediction_probs).transpose()
train_out_pred_prob_df.columns = data_type_options
test_out_pred_prob_df = pd.DataFrame(test_out_prediction_probs).transpose()
test_out_pred_prob_df.columns = data_type_options
sp_test_out_pred_prob_df = pd.DataFrame(sp_test_out_prediction_probs).transpose()
sp_test_out_pred_prob_df.columns = data_type_options
tr_test_out_pred_prob_df = pd.DataFrame(tr_test_out_prediction_probs).transpose()
tr_test_out_pred_prob_df.columns = data_type_options

output_train_filename="Results/LGBM_train_metrics.csv"
output_test_filename = "Results/LGBM_test_metrics.csv"
output_sp_test_filename = "Results/LGBM_sp_test_metrics.csv"
output_tr_test_filename = "Results/LGBM_tr_test_metrics.csv"

out_train_predict_filename="Results/LGBM_train_predictions.csv"
out_test_predict_filename="Results/LGBM_test_predictions.csv"
out_sp_test_predict_filename="Results/LGBM_sp_test_predictions.csv"
out_tr_test_predict_filename="Results/LGBM_tr_test_predictions.csv"

out_train_predict_prob_filename="Results/LGBM_train_prediction_probs.csv"
out_test_predict_prob_filename="Results/LGBM_test_prediction_probs.csv"
out_sp_predict_prob_filename="Results/LGBM_sp_prediction_probs.csv"
out_tr_predict_prob_filename="Results/LGBM_tr_prediction_probs.csv"

#Write outputs
train_output_df.to_csv(output_train_filename,index=False)
test_output_df.to_csv(output_test_filename,index=False)
sp_test_output_df.to_csv(output_sp_test_filename,index=False)
tr_test_output_df.to_csv(output_tr_test_filename,index=False)

train_output_pred_df.to_csv(out_train_predict_filename,index=False)
test_output_pred_df.to_csv(out_test_predict_filename,index=False)
sp_test_output_pred_df.to_csv(out_sp_test_predict_filename,index=False)
tr_test_output_pred_df.to_csv(out_tr_test_predict_filename,index=False)

train_out_pred_prob_df.to_csv(out_train_predict_prob_filename, index=False)
test_out_pred_prob_df.to_csv(out_test_predict_prob_filename, index=False)
sp_test_out_pred_prob_df.to_csv(out_sp_predict_prob_filename, index=False)
tr_test_out_pred_prob_df.to_csv(out_tr_predict_prob_filename, index=False)

# %%
