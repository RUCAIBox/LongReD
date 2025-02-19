
accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train_longred_uniform.py \
--batch-size 4 \
--batch-size2 128 \
--batch-size3 16 \
--gradient-accumulate-every 16 \
--output-dir ./output/llama3_longred_32k \
--wandb EasyContext \
--seed 2024 \
--max-train-steps 512  \
--learning-rate 2e-5  \
--dataset ./slim_128k_2b \
--dataset3 ./slim_8k_2b \
--dataset2 ./data_1k_new \
--model ./models/Meta-Llama-3-8B  \
--seq-length 32768 \
--rope-theta 20000000 \
--parallel_mode zigzag_ring_attn


