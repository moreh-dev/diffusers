###textual_inversion.sh PSEUDO###
output_dir=outputs
current_dir=$(pwd)

while getopts t:m:b:o:l: flag
do
    case "${flag}" in
        t) env_name=${OPTARG};;
        m) model_name=${OPTARG};;
        b) batch_size=${OPTARG};;
        o) output_dir=${OPTARG};;
        l) log_dir=${OPTARG};;
    esac
done

# Check if log_dir is provided , if not use default value of current_dir/logs/${model_name}.json
if [ -z "$log_dir" ]
then
    log_dir=${current_dir}/logs/${model_name}.json
    mkdir -p ${current_dir}/logs
else
    normalized_model_name=${model_name#*/}
    log_dir="log_terminal/${env_name}/${normalized_model_name}.log"
fi

# Check if $batch_size is provided , if not use default value of 1
if expr "$batch_size" + 0 > /dev/null 2>&1; then
  batch_size=$batch_size
else
  batch_size=9
fi

# Check if data is downloaded
data_dir=/nas/common_data/huggingface/textual_inversion/cat
echo "# ========================================================= #"
if [ -d "$data_dir" ] && [ "$(ls -A $data_dir)" ]; then
  echo "data is already in $data_dir"
else
  echo "downloading data.."
  conda run -n ${env_name} python3 download_data.py
fi

# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."
cd ..
conda run -n ${env_name} python3 ../examples/textual_inversion/textual_inversion_mlflow.py \
  --pretrained_model_name_or_path ${model_name} \
  --train_data_dir $data_dir \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size ${batch_size} \
  --gradient_accumulation_steps=4 \
  --max_train_steps=100 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir ${output_dir} \
  --log_dir ${log_dir} \
  --logging_steps=30 \