## Benchmarking Open PLM available via TRILL for Protein Crystallization Prediction

The goal of the project is to benchmark open-source protein language models (PLM) available through TRILL for:
1. Protein crystallization prediction using raw protein sequences as input.
2. Show power of protein language model which can fit on a single 48Gb GPU for a downstream protein property prediction task.
3. Learn vector representations for proteins for each PLM in a zero-shot framework without fine-tuning.
4. Fine-tuning some PLMs (such as ESM2 - 3 billion parameters) not possible even after freezing all layers except last layer with a batch size of 2 on a 48Gb GPU.
5. To have fair comparison of embedding representations learnt through PLM a zero-shot learning framework is utilized.
6. Linear probing performed on top of feature representations using optimized LightGBM and XGBoost models for the task of distinguishing crystallizable proteins from non-crystallizable ones.

 
