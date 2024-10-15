# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: TRILL
#     language: python
#     name: trill
# ---

# +
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble._forest import RandomForestClassifier
from evaluate import load
import pandas as pd

f1_metric,roc_metric,acc_metric,mcc_metric,prec_metric,rec_metric = load("f1"),load("roc_auc"),load("accuracy"),load("matthews_correlation"),load("precision"),load("recall")


# -

def get_aupr_score(y_test, predictions, method="MLPClassifier"):
    precision, recall, _ = metrics.precision_recall_curve(y_test, predictions[[method]])
    aupr = round(metrics.average_precision_score(y_test, predictions[[method]]), 3)
    return(aupr)


train_options = ["../Results/crystallization_Ankh-Large_AVG.csv",
                 "../Results/crystallization_Ankh_AVG.csv",
                 "../Results/crystallization_ProstT5_AVG.csv",
                 "../Results/crystallization_ProtT5-XL_AVG.csv",
                 "../Results/crystallization_esm2_t12_35M_AVG.csv",
                 "../Results/crystallization_esm2_t30_150M_AVG.csv",
                 "../Results/crystallization_esm2_t33_650M_AVG.csv",
                 "../Results/crystallization_esm2_t36_3B_AVG.csv",
                 "../Results/crystallization_esm2_t6_8M_AVG.csv",
                 "../Results/crystallization_xTrimoPGLM_1B_AVG.csv",
                 "../Results/crystallization_xTrimoPGLM_3B_AVG.csv",
                 "../Results/crystallization_xTrimoPGLM_10B_AVG.csv",
                 "../Results/crystallization_SaProt_35M_AVG.csv",
                 "../Results/crystallization_SaProt_650M_AVG.csv"]
sp_test_options = ["../Results/crystallization_sp_Ankh-Large_AVG.csv",
                "../Results/crystallization_sp_Ankh_AVG.csv",
                "../Results/crystallization_sp_ProstT5_AVG.csv",
                "../Results/crystallization_sp_ProtT5-XL_AVG.csv",
                "../Results/crystallization_sp_esm2_t12_35M_AVG.csv",
                "../Results/crystallization_sp_esm2_t30_150M_AVG.csv",
                "../Results/crystallization_sp_esm2_t33_650M_AVG.csv",
                "../Results/crystallization_sp_esm2_t36_3B_AVG.csv",
                "../Results/crystallization_sp_esm2_t6_8M_AVG.csv",
                "../Results/crystallization_sp_xTrimoPGLM_1B_AVG.csv",
                "../Results/crystallization_sp_xTrimoPGLM_3B_AVG.csv",
                "../Results/crystallization_sp_xTrimoPGLM_10B_AVG.csv",
                "../Results/crystallization_sp_SaProt_35M_AVG.csv",
                "../Results/crystallization_sp_SaProt_650M_AVG.csv"]
tr_test_options = ["../Results/crystallization_tr_Ankh-Large_AVG.csv",
                   "../Results/crystallization_tr_Ankh_AVG.csv",
                   "../Results/crystallization_tr_ProstT5_AVG.csv",
                   "../Results/crystallization_tr_ProtT5-XL_AVG.csv",
                   "../Results/crystallization_tr_esm2_t12_35M_AVG.csv",
                   "../Results/crystallization_tr_esm2_t30_150M_AVG.csv",
                   "../Results/crystallization_tr_esm2_t33_650M_AVG.csv",
                   "../Results/crystallization_tr_esm2_t36_3B_AVG.csv",
                   "../Results/crystallization_tr_esm2_t6_8M_AVG.csv",
                   "../Results/crystallization_tr_xTrimoPGLM_1B_AVG.csv",
                   "../Results/crystallization_tr_xTrimoPGLM_3B_AVG.csv",
                   "../Results/crystallization_tr_xTrimoPGLM_10B_AVG.csv",
                   "../Results/crystallization_tr_SaProt_35M_AVG.csv",
                   "../Results/crystallization_tr_SaProt_650M_AVG.csv"]
test_options = ["../Results/crystallization_test_Ankh-Large_AVG.csv",
                "../Results/crystallization_test_Ankh_AVG.csv",
                "../Results/crystallization_test_ProstT5_AVG.csv",
                "../Results/crystallization_test_ProtT5-XL_AVG.csv",
                "../Results/crystallization_test_esm2_t12_35M_AVG.csv",
                "../Results/crystallization_test_esm2_t30_150M_AVG.csv",
                "../Results/crystallization_test_esm2_t33_650M_AVG.csv",
                "../Results/crystallization_test_esm2_t36_3B_AVG.csv",
                "../Results/crystallization_test_esm2_t6_8M_AVG.csv",
                "../Results/crystallization_test_xTrimoPGLM_1B_AVG.csv",
                "../Results/crystallization_test_xTrimoPGLM_3B_AVG.csv",
                "../Results/crystallization_test_xTrimoPGLM_10B_AVG.csv",
                "../Results/crystallization_test_SaProt_35M_AVG.csv",
                "../Results/crystallization_test_SaProt_650M_AVG.csv"]
data_type_options = ["Ankh-Large","Ankh","ProstT5","ProtT5-XL","esm2_t12_35M","esm2_t30_150M","esm2_t33_650M","esm2_t36_3B","esm2_t6_8M","xTrimoPGLM_1B","xTrimoPGLM_3B","xTrimoPGLM_10B","SaProt_35M","SaProt_650M"]


def full_compute_metrics(predictions,labels):
    #predictions = np.argmax(predictions, axis=1)
    return([f1_metric.compute(predictions=predictions, references=labels),\
            roc_metric.compute(prediction_scores=predictions, references=labels),\
            acc_metric.compute(predictions=predictions, references=labels),\
            mcc_metric.compute(predictions=predictions, references=labels),\
            prec_metric.compute(predictions=predictions, references=labels),\
            rec_metric.compute(predictions=predictions, references=labels)])



#Read the train, test, sp and tr labels
train_labels = pd.read_csv("../Data/Crystallization/Train_True_Labels.csv",header=None)
train_labels = train_labels.iloc[:,0].astype(int).tolist()
test_labels = pd.read_csv("../Data/Crystallization/y_test.csv",header=None)
test_labels = test_labels.iloc[:,0].astype(int).tolist()
sp_test_labels = pd.read_csv("../Data/Crystallization/SP_True_Label.csv",header=None)
sp_test_labels = sp_test_labels.iloc[:,0].astype(int).tolist()
tr_test_labels = pd.read_csv("../Data/Crystallization/TR_True_Label.csv",header=None)
tr_test_labels = tr_test_labels.iloc[:,0].astype(int).tolist()

#Perform the training for each PLM and test on different test sets
model_names = ["MLPClassifier","RandomForestClassifier"]
test_output, sp_test_output, tr_test_output = [], [], []
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

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True, classifiers=[MLPClassifier, RandomForestClassifier])
    scores_df, predictions_df, models = clf.fit(X_train, X_test, train_labels, test_labels)

    #Add aupr to the results
    aupr_list = [get_aupr_score(test_labels, predictions_df, "MLPClassifier"), get_aupr_score(test_labels, predictions_df, "RandomForestClassifier")]
    scores_df["AUPR Score"] = aupr_list
    scores_df["Method"] = data_type_options[i]

    test_output.append(scores_df)

    #Get the predictions for sp test set
    sp_test_list = []
    for model_name in model_names:
        model = models[model_name]
        temp_pred = model.predict(X_sp_test)
        temp_pred_proba = model.predict(X_sp_test)

        precision, recall, _ = metrics.precision_recall_curve(sp_test_labels, temp_pred_proba)
        aupr = round(metrics.average_precision_score(sp_test_labels, temp_pred_proba), 3)

        out_sp_test = full_compute_metrics(temp_pred, sp_test_labels)
        out_sp_test_dict = dict(map(dict.popitem,out_sp_test))
        out_sp_test_dict["aupr"] = aupr
        out_sp_test_dict["Method"]=data_type_options[i]
        out_sp_test_dict["Model"]=model_name

        sp_test_list.append(out_sp_test_dict)

    sp_test_output.append(pd.DataFrame(sp_test_list))


    #Get the predictions for tr test set
    tr_test_list = []
    for model_name in model_names:
        model = models[model_name]
        temp_pred = model.predict(X_tr_test)
        temp_pred_proba = model.predict(X_tr_test)

        precision, recall, _ = metrics.precision_recall_curve(tr_test_labels, temp_pred_proba)
        aupr = round(metrics.average_precision_score(tr_test_labels, temp_pred_proba), 3)

        out_tr_test = full_compute_metrics(temp_pred, tr_test_labels)
        out_tr_test_dict = dict(map(dict.popitem,out_tr_test))
        out_tr_test_dict["aupr"] = aupr
        out_tr_test_dict["Method"]=data_type_options[i]
        out_tr_test_dict["Model"]=model_name

        tr_test_list.append(out_tr_test_dict)

    tr_test_output.append(pd.DataFrame(tr_test_list))

#Combine all the results
final_test_output = pd.concat(test_output)
final_sp_test_output = pd.concat(sp_test_output)
final_tr_test_output = pd.concat(tr_test_output)

# +
import numpy as np
#Final test output
rev_final_test_output = final_test_output.drop(columns=["Balanced Accuracy","Time Taken"])
rev_final_test_output["Model"] = rev_final_test_output.index
rev_final_test_output = rev_final_test_output[["Method","Model","F1 Score","Accuracy","MCC Score","PREC Score","REC Score","AUPR Score","ROC AUC"]]
rev_final_test_output["F1 Score"] = rev_final_test_output["F1 Score"].round(3)
rev_final_test_output["Accuracy"] = rev_final_test_output["Accuracy"].round(3)
rev_final_test_output["MCC Score"] = rev_final_test_output["MCC Score"].round(3)
rev_final_test_output["PREC Score"] = rev_final_test_output["PREC Score"].round(3)
rev_final_test_output["REC Score"] = rev_final_test_output["REC Score"].round(3)
rev_final_test_output["AUPR Score"] = rev_final_test_output["AUPR Score"].round(3)
rev_final_test_output["ROC AUC"] = rev_final_test_output["ROC AUC"].round(3)
rev_subset_final_test_output = rev_final_test_output.loc[rev_final_test_output["Model"]=="MLPClassifier",:]

rev_subset_final_test_output.to_csv("../Results/MLPClassifier_test_performance.csv",index=None)

# +
#Final sp test output
rev_final_sp_test_output = final_sp_test_output[["Method","Model","f1","accuracy","matthews_correlation","precision","recall","aupr","roc_auc"]]
rev_final_sp_test_output["f1"] = rev_final_sp_test_output["f1"].round(3)
rev_final_sp_test_output["accuracy"] = rev_final_sp_test_output["accuracy"].round(3)
rev_final_sp_test_output["matthews_correlation"] = rev_final_sp_test_output["matthews_correlation"].round(3)
rev_final_sp_test_output["precision"] = rev_final_sp_test_output["precision"].round(3)
rev_final_sp_test_output["recall"] = rev_final_sp_test_output["recall"].round(3)
rev_final_sp_test_output["aupr"] = rev_final_sp_test_output["aupr"].round(3)
rev_final_sp_test_output["roc_auc"] = rev_final_sp_test_output["roc_auc"].round(3)
rev_subset_final_sp_test_output = rev_final_sp_test_output.loc[rev_final_sp_test_output["Model"]=="MLPClassifier",:]


rev_subset_final_sp_test_output.to_csv("../Results/MLPClassifier_sp_test_performance.csv",index=None)

# +
#Final tr test output
rev_final_tr_test_output = final_tr_test_output[["Method","Model","f1","accuracy","matthews_correlation","precision","recall","aupr","roc_auc"]]
rev_final_tr_test_output["f1"] = rev_final_tr_test_output["f1"].round(3)
rev_final_tr_test_output["accuracy"] = rev_final_tr_test_output["accuracy"].round(3)
rev_final_tr_test_output["matthews_correlation"] = rev_final_tr_test_output["matthews_correlation"].round(3)
rev_final_tr_test_output["precision"] = rev_final_tr_test_output["precision"].round(3)
rev_final_tr_test_output["recall"] = rev_final_tr_test_output["recall"].round(3)
rev_final_tr_test_output["aupr"] = rev_final_tr_test_output["aupr"].round(3)
rev_final_tr_test_output["roc_auc"] = rev_final_tr_test_output["roc_auc"].round(3)
rev_subset_final_tr_test_output = rev_final_tr_test_output.loc[rev_final_tr_test_output["Model"]=="MLPClassifier",:]


rev_subset_final_tr_test_output.to_csv("../Results/MLPClassifier_tr_test_performance.csv",index=None)
# -


