
accelerate launch \
--config_file  accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 4 \
--gradient-accumulate-every 16 \
--output-dir ./output/llama3_longcpt_32k \
--wandb EasyContext \
--seed 2024 \
--max-train-steps 512  \
--learning-rate 2e-5  \
--dataset ./slim_128k_2b \
--model ./models/Meta-Llama-3-8B  \
--seq-length 32768 \
--rope-theta 20000000 \
--parallel_mode zigzag_ring_attn


