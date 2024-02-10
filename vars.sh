# Silva Datasets
declare -A datasets=(
    [silva]=silva
    [silva-filtered]=silva_filtered
    [silva-filtered-515f-806r]=silva_filtered_515f_806r
    [silva-nr99]=silva_nr99
    [silva-nr99-filtered]=silva_nr99_filtered
    [silva-nr99-filtered-515f-806r]=silva_nr99_filtered_515f_806r

    [hopland]=hopland
    [nachusa]=nachusa
    [sfd]=sfd
    [wetland]=wetland
)

# DNABERT
declare -A dnabert_pretrain_artifacts=(
    [silva]=sirdavidludwig/model-registry/dnabert-pretrain-64d-150bp:silva
    [silva-filtered]=sirdavidludwig/model-registry/dnabert-pretrain-64d-150bp:silva-filtered
    [silva-filtered-515f-806r]=sirdavidludwig/model-registry/dnabert-pretrain-64d-150bp:silva-filtered-515f-806r
    [silva-nr99]=sirdavidludwig/model-registry/dnabert-pretrain-64d-150bp:silva-nr99
    [silva-nr99-filtered]=sirdavidludwig/model-registry/dnabert-pretrain-64d-150bp:silva-nr99-filtered
    [silva-nr99-filtered-515f-806r]=sirdavidludwig/model-registry/dnabert-pretrain-64d-150bp:silva-nr99-filtered-515f-806r
    [deep-silva-nr99-filtered-515f-806r]=sirdavidludwig/dnabert-pretrain/dnabert-pretrain-deep-silva-nr99-filtered-515f-806r-64d-150bp:v1
)

# DNABERT Taxonomy
declare -A dnabert_taxonomy_artifacts=(
    [bertax-silva-nr99-filtered-515f-806r]=sirdavidludwig/taxonomy-classification/bertax-silva-nr99-filtered-515f-806r-64d-150bp:v3
    [naive-silva-nr99-filtered-515f-806r]=sirdavidludwig/taxonomy-classification/naive-silva-nr99-filtered-515f-806r-64d-150bp:v1
    [topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/taxonomy-classification/topdown-silva-nr99-filtered-515f-806r-64d-150bp:v2
)

# SetBERT
declare -A setbert_pretrain_artifacts=(
    [topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/setbert-pretrain/topdown-silva-nr99-filtered-515f-806r-64d-150l:all
    [hns-topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/setbert-pretrain/topdown-silva-nr99-filtered-515f-806r-64d-150l:hns # Leave-out Wetland
    [hnw-topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/setbert-pretrain/topdown-silva-nr99-filtered-515f-806r-64d-150l:hnw # Leave-out SFD
    [hsw-topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/setbert-pretrain/topdown-silva-nr99-filtered-515f-806r-64d-150l:hsw # Leave-out Nachusa
    [nsw-topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/setbert-pretrain/topdown-silva-nr99-filtered-515f-806r-64d-150l:nsw # Leave-out Hopland
)

# SetBERT Taxonomy Models
declare -A setbert_taxonomy_artifacts=(
    [topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/taxonomy-classification/setbert-taxonomy-topdown-silva-nr99-filtered-515f-806r-64d-150bp:v0
    [hns-topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/taxonomy-classification/setbert-taxonomy-topdown-hns-silva-nr99-filtered-515f-806r-64d-150bp:v0 # Leave-out Wetland
    [hnw-topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/taxonomy-classification/setbert-taxonomy-topdown-hnw-silva-nr99-filtered-515f-806r-64d-150bp:v0 # Leave-out SFD
    [hsw-topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/taxonomy-classification/setbert-taxonomy-topdown-hsw-silva-nr99-filtered-515f-806r-64d-150bp:v0 # Leave-out Nachusa
    [nsw-topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/taxonomy-classification/setbert-taxonomy-topdown-nsw-silva-nr99-filtered-515f-806r-64d-150bp:v0 # Leave-out Hopland
)

# SetBERT Hopland Models
declare -A setbert_hopland_artifacts=(
    [qiime-silva-nr99-filtered-515f-806r]=sirdavidludwig/hopland/setbert-hopland-qiime-silva-nr99-filtered-515f-806r-64d-150bp:v0
    [topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/hopland/setbert-hopland-qiime-br-classifier-64d-150bp:v3
)

# SetBERT SFD Models
declare -A setbert_sfd_artifacts=(
    [topdown-silva-nr99-filtered-515f-806r]=sirdavidludwig/sfd/setbert-sfd-classifier-silva-nr99-filtered-515f-806r-64d-150bp:v0
)
