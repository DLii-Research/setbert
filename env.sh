# Synthetic Data Directory
synthetic_data_path=~/work/Datasets/Synthetic

# Models -------------------------------------------------------------------------------------------

# DNABERT
export dnabert_pretrain_silva=sirdavidludwig/model-registry/dnabert-pretrain-silva-64d-150l:v0
export dnabert_taxonomy_naive=sirdavidludwig/dnabert-taxonomy/dnabert-taxonomy-naive-64d-150l:v0
export dnabert_taxonomy_bertax=sirdavidludwig/dnabert-taxonomy/dnabert-taxonomy-bertax-64d-150l:v0
export dnabert_taxonomy_topdown=sirdavidludwig/dnabert-taxonomy/dnabert-taxonomy-topdown-64d-150l:v0

# DNABERT (deeper)
export dnabert_taxonomy_topdown_deep=sirdavidludwig/dnabert-taxonomy/dnabert-taxonomy-topdown-deep-64d-150l:v0

# SetBERT
export setbert_pretrain_silva=sirdavidludwig/model-registry/setbert-pretrain-silva-64d-150l:v0
export setbert_taxonomy_topdown=sirdavidludwig/model-registry/setbert-taxonomy-topdown-64d-150l:v0

# SetBERT (leave-one-out controls)
export setbert_taxonomy_topdown_nhs=sirdavidludwig/setbert-taxonomy/setbert-taxonomy-topdown-nhs-64d-150l:v0
export setbert_taxonomy_topdown_nhw=sirdavidludwig/setbert-taxonomy/setbert-taxonomy-topdown-nhw-64d-150l:v0
export setbert_taxonomy_topdown_nsw=sirdavidludwig/setbert-taxonomy/setbert-taxonomy-topdown-nsw-64d-150l:v0
export setbert_taxonomy_topdown_hsw=sirdavidludwig/setbert-taxonomy/setbert-taxonomy-topdown-hsw-64d-150l:v0
