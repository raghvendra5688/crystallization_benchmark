## Benchmarking Open PLM available via TRILL for Protein Crystallization Prediction

The goal of the project is to benchmark open-source protein language models (PLM) available through TRILL for:
1. Protein crystallization prediction using raw protein sequences as input.
2. Show power of protein language model which can fit on a single 48Gb GPU for a downstream protein property prediction task.
3. Learn vector representations for proteins for each PLM in a zero-shot framework without fine-tuning. Fine-tuning some PLMs (such as ESM2 - 3 billion parameters) is not possible even after freezing all layers except last layer with a batch size of 2 on a 48Gb GPU.
5. To have fair comparison of vector representations learnt through a PLM, a zero-shot learning framework is utilized.
6. Linear probing performed on top of feature representations using optimized LightGBM and XGBoost models for the task of distinguishing crystallizable proteins from non-crystallizable ones.

## Data
Crystallization data in the Data/Crystallization folder including:
1. Training Set: crystallization_train.fasta, Train_True_Labels.csv, train_key.csv 
2. Test Set: crystallization_test.fasta, y_test.csv, test_key.csv
3. SP Test Set: FULL_SP.fasta, SP_True_Label.csv, sp_key.csv
4. TR Test Set: FULL_TR.fasta, TR_True_Label.csv, tr_key.csv
5. Dataset for generating crystallizable proteins: positive_crystallization_train.fasta

## Environment
Follow the instructions at `https://trill.readthedocs.io/en/latest/home.html` to install TRILL, specifically:

1. Set-up micromamba: \
`"${SHELL}" <(curl -L micro.mamba.pm/install.sh)`

2. Once micromamba is ready, set-up pre-requisties for TRILL \
`micromamba create -n TRILL python=3.10 ; micromamba activate TRILL` \
`micromamba install -c pytorch -c nvidia pytorch=2.1.2 pytorch-cuda=12.1 torchdata` \
`micromamba install -c conda-forge openbabel pdbfixer swig openmm smina fpocket vina openff-toolkit openmmforcefields setuptools=69.5.1` \
`micromamba install -c bioconda foldseek pyrsistent` \
`micromamba install -c "dglteam/label/cu121" dgl` \
`micromamba install -c pyg pyg pytorch-cluster pytorch-sparse pytorch-scatter` \
`pip install git+https://github.com/martinez-zacharya/lightdock.git@03a8bc4888c0ff8c98b7f0df4b3c671e3dbf3b1f` `git+https://github.com/martinez-zacharya/ECPICK.git setuptools==69.5.1`

3. Next install TRILL \
`pip install trill-proteins`

4. Install additional packages: \
`pip install seaborn` \
`pip install scikit-learn` \ 
`pip install scipy`

This will set-up the environment and now we are within the TRILL environment

## Scripts
The user is expected to follow the instructions below and generate the results in respective folders as not all results are deposited in the github.

1. `run_train_embed.sh`: Run the script with `bash run_train_embed.sh` \
    **Input**: Path to the training fasta file needs to be specified in the script. \
    **Output**: Will generate feature vector representation for training set using PLMs such as ESM, Ankh, ProstT5 available through TRILL in the `Results` directory. 

2. `run_train_xgboost_classifier.sh`: Run the script with `bash run_train_xgboost_classifier.sh`. \
    **Input**: Feature vector representation and training label keys to be specified in the script. \
    **Output**: 2d-UMAP visualization of proteins for each PLM vector representation in the `Results` folder and 10-fold cross-validated XGBoost model in `Models/XGBoost` folder. \
    **Hyper-parameters**: 75% of data used for traiing, no of estimators set to 200 and weighted F1 metric used to handle class imbalance in training set.

3. `run_test_embed.sh`: Run the script with `bash run_test_embed.sh`. \
    **Input**: Path to test fasta file and test key file to be specified in the script. \
    **Output**: Feature vectors are generated for test set and placed in `Results` folder along with 2D-UMAP visualizations for each PLM.

4. `run_test_sp_embed.sh`: Run the script with `bash run_test_sp_embed.sh`. \
    **Input**: Path to SP test fasta file and SP test key file to be specified in the script. \
    **Output**: Feature vectors are generated for SP test set and placed in `Results` folder along with 2D-UMAP visualizations for each PLM.


5. `run_test_tr_embed.sh`: Run the script with `bash run_test_tr_embed.sh`. \
    **Input**: Path to TR test fasta file and TR test key file to be specified in the script. \
    **Output**: Feature vectors are generated for TR test set and placed in `Results` folder along with 2D-UMAP visualizations for each PLM.

6. `run_xgboost_all_test_predictions.sh`: Run the script with `run_xgboost_all_test_predictions.sh`. \
    **Input**: Path to test fasta, feature vector representations for test set, path to XGBoost model and test set label key. \
    **Output**: XGBoost class predictions, XGBoost logits [probability] prediction and a log file in `Results/XGBoost' folder.


7. `fb_lgbm_crystal.py`: Run the script with `python fb_lgbm_crystal.py`. \
    **Input**: Path to feature representations for train and test sets for each PLM.  \
    **Output**: Cross-validated LightGBM models for each PLM with optimized hyper-parameters and test predictions for each test set in the `Results` folder. \
    **Hyper-parameters**: A grid of hyper-parameters including `n_estimators`, `max_depth`, `num_leaves`, `min_child_samples`, `learning_rate`, `subsample`, `colsample_by_tree`, `reg_alpha` and `reg_lambda` is provided in the script.

9. `gen_embed_xgb_classify_cry_proteins.sh`: Run the script with `bash gen_embed_xgb_classify_cry_proteins.sh`. \
    **Input**: Path to crystallizable protein fasta file used for fine-tuning ProtGPT2. \
    **Output**: a) Fine-tuned ProtGPT2 model for crystallization class \
                b) Set of 3000 synthetic proteins generated through fine-tuned ProtGPT2 model \
                c) Feature vector representations for these proteins through each PLM and deposited in `Results` folder \
                d) XGBoost predictions for these proteins for each PLM in the `Results/XGBoost` folder \
                e) 2D-UMAP representations of these proteins along with the labels predicted by individual XGBoost classifier.

10. `combine_generated_proteins.py`: Run the script with `python combine_generated_proteins.py`. \
   **Input**: XGBoost predictions for each PLM for the generated proteins. \
   **Output**: A dataframe with the consensus of the predictions of all the PLM models and final label assignment for each protein deposited in the `Results` folder.

 
