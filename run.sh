# construct graph
python construct_graph_coref.py --raw_dir data_GROVER/grover_test.jsonl

# baseline
CUDA_VISIBLE_DEVICES=0 python main.py \
--loss_type normal \
--data_dir data_GROVER \
--num_train_epochs 2 \
--output_dir experiments/GROVER_s10_baseline \
--train_file grover_10000_train_coref.jsonl \
--dev_file grover_dev_coref.jsonl \
--test_file grover_test_coref.jsonl \
--dataset_name GROVER

CUDA_VISIBLE_DEVICES=0 python main.py \
--loss_type normal \
--data_dir data_GPT2 \
--num_train_epochs 2 \
--output_dir experiments/GPT2_s10_baseline \
--train_file gpt2_50000_train_coref.jsonl \
--dev_file gpt2_dev_coref.jsonl \
--test_file gpt2_test_coref.jsonl \
--dataset_name GPT2

CUDA_VISIBLE_DEVICES=0 python main.py \
--loss_type normal \
--data_dir data_ChatGPT \
--num_train_epochs 2 \
--output_dir experiments/ChatGPT_s10_baseline \
--train_file train_coref.jsonl \
--dev_file valid_coref.jsonl \
--test_file test_coref.jsonl \
--dataset_name ChatGPT