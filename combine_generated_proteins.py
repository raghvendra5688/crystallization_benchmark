# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: ProtSol
#     language: python
#     name: protsol
# ---

from Bio import SeqIO
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


#Define functions
def convert_score_to_prob(df):
    df.iloc[:,0] = 1/(1+np.exp(-df.iloc[:,0]))
    df.iloc[:,1] = 1/(1+np.exp(-df.iloc[:,1]))
    total = df.iloc[:,0]+df.iloc[:,1]
    df.iloc[:,0] = df.iloc[:,0]/total
    df.iloc[:,1] = df.iloc[:,1]/total
    return(df)


# +
##Read all the XGBoost predictions
esm2_t6_8m_predictions = pd.read_csv("Results/XGBoost/esm2_t6_8M_gen_crystal_XGBoost_class_probs.csv",header='infer')
esm2_t6_8m_probs = convert_score_to_prob(esm2_t6_8m_predictions)
esm2_t6_8m_class1_probs = esm2_t6_8m_probs.iloc[:,1].tolist()

esm2_t6_8m_prediction_labels = pd.read_csv("Results//XGBoost/esm2_t6_8M_gen_crystal_XGBoost_predictions.csv",header="infer")
esm2_t6_8m_class_labels = esm2_t6_8m_prediction_labels["Class"].tolist()


#Perform t12 35m
esm2_t12_35m_predictions = pd.read_csv("Results/XGBoost/esm2_t12_35M_gen_crystal_XGBoost_class_probs.csv",header='infer')
esm2_t12_35m_probs = convert_score_to_prob(esm2_t12_35m_predictions)
esm2_t12_35m_class1_probs = esm2_t12_35m_probs.iloc[:,1].tolist()

esm2_t12_35m_prediction_labels = pd.read_csv("Results/XGBoost/esm2_t12_35M_gen_crystal_XGBoost_predictions.csv",header="infer")
esm2_t12_35m_class_labels = esm2_t12_35m_prediction_labels["Class"].tolist()

#Perform t30 150m
esm2_t30_150m_predictions = pd.read_csv("Results/XGBoost/esm2_t30_150M_gen_crystal_XGBoost_class_probs.csv",header='infer')
esm2_t30_150m_probs = convert_score_to_prob(esm2_t30_150m_predictions)
esm2_t30_150m_class1_probs = esm2_t30_150m_probs.iloc[:,1].tolist()

esm2_t30_150m_prediction_labels = pd.read_csv("Results/XGBoost/esm2_t30_150M_gen_crystal_XGBoost_predictions.csv",header="infer")
esm2_t30_150m_class_labels = esm2_t30_150m_prediction_labels["Class"].tolist()

#Perform t33 650m
esm2_t33_650m_predictions = pd.read_csv("Results/XGBoost/esm2_t30_150M_gen_crystal_XGBoost_class_probs.csv",header='infer')
esm2_t33_650m_probs = convert_score_to_prob(esm2_t30_150m_predictions)
esm2_t33_650m_class1_probs = esm2_t30_150m_probs.iloc[:,1].tolist()

esm2_t33_650m_prediction_labels = pd.read_csv("Results/XGBoost/esm2_t30_150M_gen_crystal_XGBoost_predictions.csv",header="infer")
esm2_t33_650m_class_labels = esm2_t30_150m_prediction_labels["Class"].tolist()

#Perform t36 3b
esm2_t36_3b_predictions = pd.read_csv("Results/XGBoost/esm2_t36_3B_gen_crystal_XGBoost_class_probs.csv",header='infer')
esm2_t36_3b_probs = convert_score_to_prob(esm2_t36_3b_predictions)
esm2_t36_3b_class1_probs = esm2_t36_3b_probs.iloc[:,1].tolist()

esm2_t36_3b_prediction_labels = pd.read_csv("Results/XGBoost/esm2_t36_3B_gen_crystal_XGBoost_predictions.csv",header="infer")
esm2_t36_3b_class_labels = esm2_t36_3b_prediction_labels["Class"].tolist()


#Perform Ankh
ankh_predictions = pd.read_csv("Results/XGBoost/Ankh_gen_crystal_XGBoost_class_probs.csv",header="infer")
ankh_probs = convert_score_to_prob(ankh_predictions)
ankh_class1_probs = ankh_probs.iloc[:,1].tolist()

ankh_prediction_labels = pd.read_csv("Results/XGBoost/Ankh_gen_crystal_XGBoost_predictions.csv",header="infer")
ankh_class_labels = ankh_prediction_labels["Class"].tolist()

#Perform Ankh-Large
ankh_large_predictions = pd.read_csv("Results/XGBoost/Ankh-Large_gen_crystal_XGBoost_class_probs.csv",header="infer")
ankh_large_probs = convert_score_to_prob(ankh_large_predictions)
ankh_large_class1_probs = ankh_large_probs.iloc[:,1].tolist()

ankh_large_prediction_labels = pd.read_csv("Results/XGBoost/Ankh-Large_gen_crystal_XGBoost_predictions.csv",header="infer")
ankh_large_class_labels = ankh_large_prediction_labels["Class"].tolist()

#Perform prost5
prostt5_predictions = pd.read_csv("Result/XGBoost/ProstT5_gen_crystal_XGBoost_class_probs.csv",header="infer")
prostt5_probs = convert_score_to_prob(prostt5_predictions)
prostt5_class1_probs = prostt5_probs.iloc[:,1].tolist()

prostt5_prediction_labels = pd.read_csv("Results/XGBoost/ProstT5_gen_crystal_XGBoost_predictions.csv",header="infer")
prostt5_class_labels = prostt5_prediction_labels["Class"].tolist()

#Perform protT5-xl
prott5_xl_predictions = pd.read_csv("Results/XGBoost/ProtT5-XL_gen_crystal_XGBoost_class_probs.csv",header="infer")
prott5_xl_probs = convert_score_to_prob(prott5_xl_predictions)
prott5_xl_class1_probs = prott5_xl_probs.iloc[:,1].tolist()

prott5_xl_prediction_labels = pd.read_csv("Results/XGBoost/ProtT5-XL_gen_crystal_XGBoost_predictions.csv",header="infer")
prott5_xl_class_labels = prott5_xl_prediction_labels["Class"].tolist()


# +
#Protein info total
protein_labels = prott5_xl_prediction_labels["Label"].tolist()

all_lists = [protein_labels,esm2_t6_8m_class_labels,esm2_t12_35m_class_labels,esm2_t30_150m_class_labels,esm2_t33_650m_class_labels,esm2_t36_3b_class_labels,ankh_class_labels,ankh_large_class_labels,prostt5_class_labels,prott5_xl_class_labels,\
             esm2_t12_35m_class1_probs,esm2_t12_35m_class1_probs,esm2_t30_150m_class1_probs,esm2_t33_650m_class1_probs,esm2_t36_3b_class1_probs,ankh_class1_probs,ankh_large_class1_probs,prostt5_class1_probs,prott5_xl_class1_probs]

revised_prediction_df = pd.DataFrame(all_lists).transpose()
revised_prediction_df.columns = ["SeqId","ESM2_T6_8M_Labels","ESM2_T12_35M_Labels","ESM2_T30_150M_Labels","ESM2_T33_650M_Labels","ESM2_T36_3B_Labels","Ankh_Labels","Ankh-Large_Labels","ProstT5_Labels","ProtT5-XL_Labels",\
                                 "ESM2_T6_8M_Class1_Prob","ESM2_T12_35M_Class1_Prob","ESM2_T30_150M_Class1_Prob","ESM2_T33_650M_Class1_Prob","ESM2_T36_3B_Class1_Prob","Ankh_Class1_Prob","Ankh-Large_Class1_Prob","ProstT5_Class1_Prob","ProtT5-XL_Class1_Prob"]
revised_prediction_df["Common_Crystallizable"]=1
revised_prediction_df["Mean_Prob"]=revised_prediction_df.iloc[:,10:19].mean(axis=1)
revised_prediction_df.head()
# -

#Find all rows with crystallizable proteins
findings_df = revised_prediction_df=="crystallizable"
common_crystallizable=findings_df.select_dtypes(include=['bool']).sum(axis=1).tolist()
final_common_crystallizable=[]
for entry in common_crystallizable:
    if entry == 9:
        final_common_crystallizable.append(1)
    else:
        final_common_crystallizable.append(0)
revised_prediction_df["Common_Crystallizable"]=final_common_crystallizable
revised_prediction_df.to_csv("/Results/Generated_Crystallizable_Proteins_Consensus.csv",index=False)

#Load the fasta file
records = list(SeqIO.parse("Data/Crystallization/gen_crystal_ProtGPT2.fasta", "fasta"))
all_seq_ids = [r.id for r in records]
all_seqs = [str(r.seq)  for r in records]
rev_seq_ids, rev_seqs=[],[]
for i in range(0,len(final_common_crystallizable)):
    if (final_common_crystallizable[i]==1):
        rev_seq_ids.append(all_seq_ids[i])
        rev_seqs.append(all_seqs[i])
fp = open("Data/Crystallization/gen_filtered_crystal_ProtGPT2.fasta","w")
for i in range(0,len(rev_seqs)):
    outputstring = ">"+rev_seq_ids[i]+"\n"+rev_seqs[i]
    fp.write(outputstring+"\n")
fp.close()

rev_seq_lens = [len(x) for x in rev_seqs]
plt.hist(rev_seq_lens)

print(len(rev_seqs))
