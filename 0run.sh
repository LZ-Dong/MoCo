CUDA_VISIBLE_DEVICES=0 python main.py \
--loss_type rfcl \
--data_dir GROVER300 \
--num_train_epochs 10 \
--output_dir experiments/GROVER300_coco_10 \
--train_file train_coref.jsonl \
--dev_file test_coref.jsonl \
--test_file test_coref.jsonl \
--dataset_name GROVER