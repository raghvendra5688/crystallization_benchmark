#!/usr/bin/bash 
##TRILL Fine-tune ProtGPT2 model
trill --outdir Models --n_workers 40 crystallization 1 finetune ProtGPT2 Data/Crystallization/positive_crystallization_train.fasta --epochs 10 
##TRILL generate proteins from the class of crystallizable proteins
trill --outdir Data/Crystallization --n_workers 40 gen_crystal 1 lang_gen ProtGPT2 --finetuned Models/crystallization_ProtGPT2_10.pt --num_return_sequences 3000 --repetition_penalty 2.0 --batch_size 8
## TRILL: Generate embeddings from fine-tuned ESM models and run the bigger models of the shelf for test data
trill --outdir Results --n_workers 40 gen_crystal 1 embed esm2_t6_8M Data/Crystallization/gen_crystal_ProtGPT2.fasta --batch_size 8 --avg 
trill --outdir Results --n_workers 40 gen_crystal 1 embed esm2_t12_35M Data/Crystallization/gen_crystal_ProtGPT2.fasta --batch_size 8 --avg
trill --outdir Results --n_workers 40 gen_crystal 1 embed esm2_t30_150M Data/Crystallization/gen_crystal_ProtGPT2.fasta --batch_size 8 --avg
trill --outdir Results --n_workers 40 gen_crystal 1 embed esm2_t33_650M Data/Crystallization/gen_crystal_ProtGPT2.fasta --batch_size 4 --avg
trill --outdir Results --n_workers 40 gen_crystal 1 embed esm2_t36_3B Data/Crystallization/gen_crystal_ProtGPT2.fasta --batch_size 4 --avg
trill --outdir Results --n_workers 40 gen_crystal 1 embed ProtT5-XL Data/Crystallization/gen_crystal_ProtGPT2.fasta --batch_size 4 --avg
trill --outdir Results --n_workers 40 gen_crystal 1 embed ProstT5 Data/Crystallization/gen_crystal_ProtGPT2.fasta --batch_size 4 --avg
trill --outdir Results --n_workers 40 gen_crystal 1 embed Ankh Data/Crystallization/gen_crystal_ProtGPT2.fasta --batch_size 1 --avg
trill --outdir Results --n_workers 40 gen_crystal 1 embed Ankh-Large Data/Crystallization/gen_crystal_ProtGPT2.fasta --batch_size 1 --avg
## TRILL: Get predictions for XGBoost on generated sequences
trill --outdir Results/XGBoost --n_workers 40 esm2_t6_8M_gen_crystal 1 classify XGBoost Data/Crystallization/gen_crystal_ProtGPT2.fasta --emb_model esm2_t6_8M --preComputed_Embs Results/gen_crystal_esm2_t6_8M_AVG.csv --preTrained Models/XGBoost/esm2_t6_8M_XGBoost_320.json
trill --outdir Results/XGBoost --n_workers 40 esm2_t12_35M_gen_crystal 1 classify XGBoost Data/Crystallization/gen_crystal_ProtGPT2.fasta --emb_model esm2_t12_35M --preComputed_Embs Results/gen_crystal_esm2_t12_35M_AVG.csv --preTrained Models/XGBoost/esm2_t12_35M_XGBoost_480.json 
trill --outdir Results/XGBoost --n_workers 40 esm2_t30_150M_gen_crystal 1 classify XGBoost Data/Crystallization/gen_crystal_ProtGPT2.fasta --emb_model esm2_t30_150M --preComputed_Embs Results/gen_crystal_esm2_t30_150M_AVG.csv --preTrained Models/XGBoost/esm2_t30_150M_XGBoost_640.json 
trill --outdir Results/XGBoost --n_workers 40 esm2_t33_650M_gen_crystal 1 classify XGBoost Data/Crystallization/gen_crystal_ProtGPT2.fasta --emb_model esm2_t33_650M --preComputed_Embs Results/gen_crystal_esm2_t33_650M_AVG.csv --preTrained Models/XGBoost/esm2_t33_650M_XGBoost_1280.json 
trill --outdir Results/XGBoost --n_workers 40 esm2_t36_3B_gen_crystal 1 classify XGBoost Data/Crystallization/gen_crystal_ProtGPT2.fasta --emb_model esm2_t36_3B --preComputed_Embs Results/gen_crystal_esm2_t36_3B_AVG.csv --preTrained Models/XGBoost/esm2_t36_3B_XGBoost_2560.json 
trill --outdir Results/XGBoost --n_workers 40 ProtT5-XL_gen_crystal 1 classify XGBoost Data/Crystallization/gen_crystal_ProtGPT2.fasta --emb_model ProtT5-XL --preComputed_Embs Results/gen_crystal_ProtT5-XL_AVG.csv --preTrained Models/XGBoost/ProtT5-XL_XGBoost_1024.json 
trill --outdir Results/XGBoost --n_workers 40 ProstT5_gen_crystal 1 classify XGBoost Data/Crystallization/gen_crystal_ProtGPT2.fasta --emb_model ProstT5 --preComputed_Embs Results/gen_crystal_ProstT5_AVG.csv --preTrained Models/XGBoost/ProstT5_XGBoost_1024.json 
trill --outdir Results/XGBoost --n_workers 40 Ankh_gen_crystal 1 classify XGBoost Data/Crystallization/gen_crystal_ProtPGT2.fasta --emb_model Ankh --preComputed_Embs Results/gen_crystal_Ankh_AVG.csv --preTrained Models/XGBoost/Ankh_XGBoost_768.json 
trill --outdir Results/XGBoost --n_workers 40 Ankh-Large_gen_crystal 1 classify XGBoost Data/Crystallization/gen_crystal_ProtGPT2.fasta --emb_model Ankh-Large --preComputed_Embs Results/gen_crystal_Ankh-Large_AVG.csv --preTrained Models/XGBoost/Ankh-Large_XGBoost_1536.json 
python replace_cols.py
## TRILL: Visualize the embeddings in 2d using the visualize functionality
trill --outdir Results --n_workers 40 vis 0 visualize Results/gen_crystal_esm2_t6_8M_AVG.csv --method UMAP --key Results/XGBoost/esm2_t6_8M_gen_crystal_XGBoost_predictions.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/gen_crystal_esm2_t12_35M_AVG.csv --method UMAP --key Results/XGBoost/esm2_t12_35M_gen_crystal_XGBoost_predictions.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/gen_crystal_esm2_t30_150M_AVG.csv --method UMAP --key Results/XGBoost/esm2_t30_150M_gen_crystal_XGBoost_predictions.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/gen_crystal_esm2_t33_650M_AVG.csv --method UMAP --key Results/XGBoost/esm2_t33_650M_gen_crystal_XGBoost_predictions.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/gen_crystal_esm2_t36_3B_AVG.csv --method UMAP --key Results/XGBoost/esm2_t36_3B_gen_crystal_XGBoost_predictions.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/gen_crystal_ProtT5-XL_AVG.csv --method UMAP --key Results/XGBoost/ProtT5-XL_gen_crystal_XGBoost_predictions.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/gen_crystal_ProstT5_AVG.csv --method UMAP --key Results/XGBoost/ProstT5_gen_crystal_XGBoost_predictions.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/gen_crystal_Ankh_AVG.csv --method UMAP --key Results/XGBoost/Ankh_gen_crystal_XGBoost_predictions.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/gen_crystal_Ankh-Large_AVG.csv --method UMAP --key Results/XGBoost/Ankh-Large_gen_crystal_XGBoost_predictions.csv
mv *.csv Results/
