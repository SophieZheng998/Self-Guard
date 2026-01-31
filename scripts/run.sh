#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
gpu=0
mem_percentage=0.60
model_name="Qwen3-8B"
model_path="Qwen/Qwen3-8B"

# extract steering vectors for contrastive dataset
dataset_path="../steering/constrastive_dataset/dataset/star1000_selfreminder.json"
output_file_name="star1000_selfreminder_steering_vector.svec"
method="diff"
accumulate_last_x_tokens=1
batch_size=16
output_dir="../steering/steering_vectors/${model_name}/${method}/"
python ../steering/extract_steering_vector.py \
    --model_path $model_path \
    --dataset_path $dataset_path \
    --output_dir $output_dir \
    --output_file_name $output_file_name \
    --method $method \
    --accumulate_last_x_tokens $accumulate_last_x_tokens \
    --batch_size $batch_size \
    --gpu $gpu \
    --mem_percentage $mem_percentage

# applied the steering vectors for model inference
is_steer=True
steer_vector_path="../steering/steering_vectors/${model_name}/diff/star1000_selfreminder_steering_vector.svec"
defense_type="star1000_selfreminder_steering_full_test"
output_dir="${model_name}/${defense_type}/"
steering_strength=0.5
steering_layer=20
steering_layer_name=20

## safety dataset
max_length=8192
dataset_name="harmbench"
dataset_id="walledai/HarmBench"
eval_model_name="Llama-Guard"
output_file_name="s${steering_strength}_l${steering_layer_name}_${dataset_name}"
python ../evaluation/safety_benchmark/safety_benchmark_selfreminder_steering.py \
    --model_name $model_path \
    --data_path $dataset_id \
    --output_dir $output_dir \
    --output_file_name $output_file_name \
    --eval_model_name $eval_model_name \
    --steer_vector_path $steer_vector_path \
    --steering_strength $steering_strength \
    --is_steer $is_steer \
    --layer_ids $steering_layer \
    --gpu $gpu \
    --mem_percentage $mem_percentage

## utility dataset
max_length=32768
eval_mode="math"
output_dir="../steering/steering_results/${model_name}/${defense_type}/s${steering_strength}_l${steering_layer_name}/"
python ../evaluation/utility_benchmark/utility_benchmark_selfreminder_steering.py \
    --model $model_name \
    --eval_mode $eval_mode \
    --max_tokens $max_length \
    --output_dir $output_dir \
    --steer_vector_path $steer_vector_path \
    --steering_strength $steering_strength \
    --is_steer $is_steer \
    --layer_ids $steering_layer \
    --gpu $gpu \
    --mem_percentage $mem_percentage

## overrefusal dataset
max_length=8192
dataset_name="xstest"
dataset_id="walledai/XSTest"
eval_model_name="wildguard"
output_file_name="s${steering_strength}_l${steering_layer_name}_${dataset_name}"
python ../evaluation/overrefusal_benchmark/overrefusal_benchmark_selfreminder_steering.py \
    --model_name $model_path \
    --data_path $dataset_id \
    --output_dir $output_dir \
    --output_file_name $output_file_name \
    --eval_model_name $eval_model_name \
    --steer_vector_path $steer_vector_path \
    --steering_strength $steering_strength \
    --is_steer $is_steer \
    --layer_ids $steering_layer \
    --gpu $gpu \
    --mem_percentage $mem_percentage