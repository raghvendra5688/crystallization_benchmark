#!/usr/bin/bash 
## TRILL: Generate embeddings from of the shelf [Zero-Shot] for embedding generation [crystallization prediction]
trill --outdir Results --n_workers 40 crystallization 1 embed esm2_t6_8M Data/Crystallization/crystallization_train.fasta --batch_size 8 --avg
trill --outdir Results --n_workers 40 crystallization 1 embed esm2_t12_35M Data/Crystallization/crystallization_train.fasta --batch_size 8 --avg
trill --outdir Results --n_workers 40 crystallization 1 embed esm2_t30_150M Data/Crystallization/crystallization_train.fasta --batch_size 8 --avg
trill --outdir Results --n_workers 40 crystallization 1 embed esm2_t33_650M Data/Crystallization/crystallization_train.fasta --batch_size 4 --avg
trill --outdir Results --n_workers 40 crystallization 1 embed esm2_t36_3B Data/Crystallization/crystallization_train.fasta --batch_size 4 --avg 
trill --outdir Results --n_workers 40 crystallization 1 embed ProtT5-XL Data/Crystallization/crystallization_train.fasta --batch_size 4 --avg 
trill --outdir Results --n_workers 40 crystallization 1 embed ProstT5 Data/Crystallization/crystallization_train.fasta --batch_size 4 --avg 
trill --outdir Results --n_workers 40 crystallization 1 embed Ankh Data/Crystallization/crystallization_train.fasta --batch_size 1 --avg
trill --outdir Results --n_workers 40 crystallization 1 embed Ankh-Large Data/Crystallization/crystallization_train.fasta --batch_size 1 --avg
