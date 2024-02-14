python gen_seq.py # construct transaction sequence
python gen_pretrain_data.py # generate pre-training data
python gen_finetune_data.py # generate fine-tuning data

python run_pretrain.py # pre-training
python run_finetune.py --init_checkpoint=ckpt_dir/model_64000 # fine-tuning and evaluation
