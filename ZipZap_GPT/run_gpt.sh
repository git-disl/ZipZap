


python gen_seq_gpt.py --phisher=True --bizdate=rebuttal_1204_nodup_nodrop
python gen_gpt_pretrain_data.py --bizdate=rebuttal_1204_nodup_nodrop_seq100 --source_bizdate=rebuttal_1204_nodup_nodrop --max_seq_length=100
python gen_gpt_finetune_phisher_data.py --bizdate=rebuttal_1204_nodup_nodrop_seq100 --source_bizdate=rebuttal_1204_nodup_nodrop --max_seq_length=100

CUDA_VISIBLE_DEVICES=1 python run_pretrain.py --bizdate=rebuttal_1204_nodup_nodrop_seq100 --max_seq_length=100 --checkpointDir=rebuttal_1204

CUDA_VISIBLE_DEVICES=2 python run_finetune_phisher.py --bizdate=rebuttal_1204_nodup_nodrop_seq100 --max_seq_length=100 --checkpointDir=tmp --init_checkpoint=rebuttal_1204/model_64000


--init_checkpoint=rebuttal_1203_exp_4bucket/model_56000

CUDA_VISIBLE_DEVICES=5 python run_pretrain.py

# train on dup transactions

python gen_seq_gpt.py --phisher=True --dup=True --bizdate=rebuttal_1204_dup

python gen_gpt_pretrain_data.py --bizdate=rebuttal_1204_dup_nodrop --source_bizdate=rebuttal_1204_dup --max_seq_length=100
python gen_gpt_finetune_phisher_data.py --bizdate=rebuttal_1204_dup_nodrop --source_bizdate=rebuttal_1204_dup --max_seq_length=100


CUDA_VISIBLE_DEVICES=1 python run_pretrain.py --bizdate=rebuttal_1204_dup_nodrop --max_seq_length=100 --checkpointDir=rebuttal_1204_dup_nodrop
