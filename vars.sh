# Silva Datasets
declare -A datasets=(
    [silva-nr99]=Silva_Nr99
    [silva-nr99-filtered]=Silva_Nr99_Filtered
    [silva-nr99-filtered-515f-806r]=Silva_Nr99_Filtered_515f_806r
)

# DNABERT
declare -A dnabert_pretrain_artifacts=(
    [silva-nr99]=sirdavidludwig/model-registry/dnabert-pretrain-64d-150bp:silva-nr99
    [silva-nr99-filtered]=sirdavidludwig/model-registry/dnabert-pretrain-64d-150bp:silva-nr99-filtered
    [silva-nr99-filtered-515f-806r]=sirdavidludwig/model-registry/dnabert-pretrain-64d-150bp:silva-nr99-filtered-515f-806r
)
