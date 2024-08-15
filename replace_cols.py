import pandas as pd
df = pd.read_csv("Results/XGBoost/esm2_t6_8M_gen_crystal_XGBoost_predictions.csv",header="infer")
rev_df = df.iloc[:,[1,0]].copy()
rev_df.columns = ["Label","Class"]
rev_df["Class"] = rev_df["Class"].replace({0: "non-crystallizable", 1: "crystallizable"})
rev_df.to_csv("Results/XGBoost/esm2_t6_8M_gen_crystal_XGBoost_predictions.csv",index=None)

df = pd.read_csv("Results/XGBoost/esm2_t12_35M_gen_crystal_XGBoost_predictions.csv",header="infer")
rev_df = df.iloc[:,[1,0]].copy()
rev_df.columns = ["Label","Class"]
rev_df["Class"] = rev_df["Class"].replace({0: "non-crystallizable", 1: "crystallizable"})
rev_df.to_csv("Results/XGBoost/esm2_t12_35M_gen_crystal_XGBoost_predictions.csv",index=None)

df = pd.read_csv("Results/XGBoost/esm2_t30_150M_gen_crystal_XGBoost_predictions.csv",header="infer")
rev_df = df.iloc[:,[1,0]].copy()
rev_df.columns = ["Label","Class"]
rev_df["Class"] = rev_df["Class"].replace({0: "non-crystallizable", 1: "crystallizable"})
rev_df.to_csv("Results/XGBoost/esm2_t30_150M_gen_crystal_XGBoost_predictions.csv",index=None)

df = pd.read_csv("Results/XGBoost/esm2_t33_650M_gen_crystal_XGBoost_predictions.csv",header="infer")
rev_df = df.iloc[:,[1,0]].copy()
rev_df.columns = ["Label","Class"]
rev_df["Class"] = rev_df["Class"].replace({0: "non-crystallizable", 1: "crystallizable"})
rev_df.to_csv("Results/XGBoost/esm2_t33_650M_gen_crystal_XGBoost_predictions.csv",index=None)

df = pd.read_csv("Results/XGBoost/esm2_t36_3B_gen_crystal_XGBoost_predictions.csv",header="infer")
rev_df = df.iloc[:,[1,0]].copy()
rev_df.columns = ["Label","Class"]
rev_df["Class"] = rev_df["Class"].replace({0: "non-crystallizable", 1: "crystallizable"})
rev_df.to_csv("Results/XGBoost/esm2_t36_3B_gen_crystal_XGBoost_predictions.csv",index=None)

df = pd.read_csv("Results/XGBoost/Ankh_gen_crystal_XGBoost_predictions.csv",header="infer")
rev_df = df.iloc[:,[1,0]].copy()
rev_df.columns = ["Label","Class"]
rev_df["Class"] = rev_df["Class"].replace({0: "non-crystallizable", 1: "crystallizable"})
rev_df.to_csv("Results/XGBoost/Ankh_gen_crystal_XGBoost_predictions.csv",index=None)

df = pd.read_csv("Results/XGBoost/Ankh-Large_gen_crystal_XGBoost_predictions.csv",header="infer")
rev_df = df.iloc[:,[1,0]].copy()
rev_df.columns = ["Label","Class"]
rev_df["Class"] = rev_df["Class"].replace({0: "non-crystallizable", 1: "crystallizable"})
rev_df.to_csv("Results/XGBoost/Ankh-Large_gen_crystal_XGBoost_predictions.csv",index=None)

df = pd.read_csv("Results/XGBoost/ProstT5_gen_crystal_XGBoost_predictions.csv",header="infer")
rev_df = df.iloc[:,[1,0]].copy()
rev_df.columns = ["Label","Class"]
rev_df["Class"] = rev_df["Class"].replace({0: "non-crystallizable", 1: "crystallizable"})
rev_df.to_csv("Results/XGBoost/ProstT5_gen_crystal_XGBoost_predictions.csv",index=None)

df = pd.read_csv("Results/XGBoost/ProtT5-XL_gen_crystal_XGBoost_predictions.csv",header="infer")
rev_df = df.iloc[:,[1,0]].copy()
rev_df.columns = ["Label","Class"]
rev_df["Class"] = rev_df["Class"].replace({0: "non-crystallizable", 1: "crystallizable"})
rev_df.to_csv("Results/XGBoost/ProtT5-XL_gen_crystal_XGBoost_predictions.csv",index=None)
