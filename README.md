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

1. Set-up micromamba:
`"${SHELL}" <(curl -L micro.mamba.pm/install.sh)`

2. Once micromamba is ready, set-up pre-requisties for TRILL
`micromamba create -n TRILL python=3.10 ; micromamba activate TRILL`
`micromamba install -c pytorch -c nvidia pytorch=2.1.2 pytorch-cuda=12.1 torchdata`
`micromamba install -c conda-forge openbabel pdbfixer swig openmm smina fpocket vina openff-toolkit openmmforcefields setuptools=69.5.1`
`micromamba install -c bioconda foldseek pyrsistent`
`micromamba install -c "dglteam/label/cu121" dgl`
`micromamba install -c pyg pyg pytorch-cluster pytorch-sparse pytorch-scatter`
`pip install git+https://github.com/martinez-zacharya/lightdock.git@03a8bc4888c0ff8c98b7f0df4b3c671e3dbf3b1f git+https://github.com/martinez-zacharya/ECPICK.git setuptools==69.5.1`

3. Next install TRILL
`pip install trill-proteins`

4. Install additional packages:
`pip install seaborn`
`pip install scikit-learn`
`pip install scipy`

This will set-up the environment and now we are within the TRILL environment
