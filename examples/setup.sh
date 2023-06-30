#!/bin/bash

base_env=$(conda info | grep -i 'base environment' | awk -F': ' '{print $2}' | sed 's/ (read only)//' | tr -d ' ')
current_dir=$(pwd)

task=$1
env_name=$task
echo "# ========================================================= #"
echo "create env for task ${task}.."

if conda env list | grep -q -E "^$env_name\s"; then
    source ${base_env}/etc/profile.d/conda.sh
    conda activate ${env_name}
else
    conda clean --all --force-pkgs-dir -y
    conda create --name ${env_name} python=3.8 -y
    source ${base_env}/etc/profile.d/conda.sh
    conda activate ${env_name}
    install_requirements=1
fi
echo "environment name: ${env_name}"

if [ "$CONDA_DEFAULT_ENV" = "${env_name}" ] && [ "$install_requirements" == "1" ]; then
    echo "installing requirements in conda env ${env_name}.."
    cd ..
    pip install -e .
    cd ${current_dir}/${task}
    pip install -r requirements.txt
    pip install mlflow
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    moreh-switch-model -M 2
    echo -e "\\n" | update-moreh --torch 1.13.1 --target 23.6.0 --force
fi

# YAML content for accelerate config
yaml_content=$(cat <<-EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: 0
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
)

# Config file path
config_file="$HOME/.cache/huggingface/accelerate/default_config.yaml"

# Create the YAML file
mkdir -p "$(dirname "$config_file")"
echo "$yaml_content" > "$config_file"

echo "YAML file created: $config_file"
