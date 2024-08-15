#!/usr/bin/bash 
## TRILL Generate embeddings from fine-tuned ESM models and run the bigger models of the shelf for TR data
trill --outdir Results --n_workers 40 crystallization_tr 1 embed esm2_t6_8M Data/Crystallization/FULL_TR.fasta --batch_size 8 --avg
trill --outdir Results --n_workers 40 crystallization_tr 1 embed esm2_t12_35M Data/Crystallization/FULL_TR.fasta --batch_size 8 --avg
trill --outdir Results --n_workers 40 crystallization_tr 1 embed esm2_t30_150M Data/Crystallization/FULL_TR.fasta --batch_size 8 --avg
trill --outdir Results --n_workers 40 crystallization_tr 1 embed esm2_t33_650M Data/Crystallization/FULL_TR.fasta --batch_size 4 --avg
trill --outdir Results --n_workers 40 crystallization_tr 1 embed esm2_t36_3B Data/Crystallization/FULL_TR.fasta --batch_size 4 --avg
trill --outdir Results --n_workers 40 crystallization_tr 1 embed ProtT5-XL Data/Crystallization/FULL_TR.fasta --batch_size 4 --avg
trill --outdir Results --n_workers 40 crystallization_tr 1 embed ProstT5 Data/Crystallization/FULL_TR.fasta --batch_size 4 --avg
trill --outdir Results --n_workers 40 crystallization_tr 1 embed Ankh Data/Crystallization/FULL_TR.fasta --batch_size 1 --avg
trill --outdir Results --n_workers 40 crystallization_tr 1 embed Ankh-Large Data/Crystallization/FULL_TR.fasta --batch_size 1 --avg
## TRILL: Visualize the embeddings in 2d using the visualize functionality
trill --outdir Results --n_workers 40 vis 0 visualize Results/crystallization_tr_esm2_t6_8M_AVG.csv --method UMAP --key Data/Crystallization/tr_key.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/crystallization_tr_esm2_t12_35M_AVG.csv --method UMAP --key Data/Crystallization/tr_key.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/crystallization_tr_esm2_t30_150M_AVG.csv --method UMAP --key Data/Crystallization/tr_key.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/crystallization_tr_esm2_t33_650M_AVG.csv --method UMAP --key Data/Crystallization/tr_key.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/crystallization_tr_esm2_t36_3B_AVG.csv --method UMAP --key Data/Crystallization/tr_key.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/crystallization_tr_ProtT5-XL_AVG.csv --method UMAP --key Data/Crystallization/tr_key.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/crystallization_tr_ProstT5_AVG.csv --method UMAP --key Data/Crystallization/tr_key.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/crystallization_tr_Ankh_AVG.csv --method UMAP --key Data/Crystallization/tr_key.csv
trill --outdir Results --n_workers 40 vis 0 visualize Results/crystallization_tr_Ankh-Large_AVG.csv --method UMAP --key Data/Crystallization/tr_key.csv
mv *.csv Results/
