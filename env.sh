# Load the .env file
set -o allexport
source .env set
set +o allexport

alias python_tf="$(echo "${python_prefix} ${python_tf}" | xargs)"
alias python_q2="$(echo "${python_prefix} conda run -n ${qiime2_env} python3" | xargs)"
