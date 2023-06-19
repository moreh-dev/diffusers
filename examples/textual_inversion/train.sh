###textual_inversion.sh PSEUDO###
env_name=${1}
model_name=${2}
batch_size=${3}
output_dir=${4:-"outputs"}
current_dir=$(pwd)
log_dir=${5:-"${current_dir}/logs/${model_name}.json"}

# Check if $batch_size is provided , if not use default value of 5
if expr "$batch_size" + 0 > /dev/null 2>&1; then
  batch_size=$batch_size
else
  batch_size=1
fi

# Check if data is downloaded
data_dir=/nas/common_data/huggingface/textual_inversion/cat
echo "# ========================================================= #"
if [ -d "$data_dir" ] && [ "$(ls -A $data_dir)" ]; then
  echo "data is already in $data_dir"
else
  echo "downloading data.."
  conda run -n ${env_name} python3 /nas/thuchk/repos/diffusers/examples/textual_inversion/download_data.py
fi

# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
conda run -n ${env_name} python3 textual_inversion.py \
  --pretrained_model_name_or_path ${model_name} \
  --train_data_dir $data_dir \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size ${batch_size} \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir ${output_dir} \