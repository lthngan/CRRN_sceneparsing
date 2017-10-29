--data_dir=../../Data/SiftFlow_Oct30_new/data_train --batch_size=32 --train_dir=./train/Nov09/Down32/SiftFlow_16_8_3_noAttWeight_shareWeight --learning_rate=0.1 --image_size=256 --use_attention_weight=False --reduce_size=32 --block_size=8 --hid_size=16 --usePredictionMax=False --database=SiftFlow --num_gpus=1 --do_val=500


Ver0.6.5 - shareweight
TRAIN
--data_dir=../../Data/SiftFlow_Oct30_new/data_train --batch_size=32 --train_dir=./train/Nov09/Down64/SiftFlow_16_8_3_noAttWeight_shareWeight --learning_rate=0.1 --image_size=256 --use_attention_weight=False --reduce_size=32 --block_size=8 --hid_size=16 --usePredictionMax=False --database=SiftFlow --num_gpus=1 --do_val=500

Ver0.6.5 - shareweight
evaluation

SIFTFLOW
--eval_dir=../../Data/SiftFlow_Oct30_new/data_eval --data_dir=../../Data/SiftFlow_Oct30_new/data_train --subset=validation --checkpoint_dir=./train/Nov09/Down32/FromServer_new/SiftFlow_8_8_3_withAttWeight_shareWeight --run_once --out_label_dir=./results/Nov09/Down32/FromServer_new/SiftFlow_8_8_3_withAttWeight_shareWeight_sumDAG_13000 --num_examples=2000 --image_size=256 --batch_size=32 --reduce_size=32 --block_size=8 --hid_size=8 --use_attention_weight=True --database=SiftFlow



ADE
--eval_dir=../../Data/ADEChallenge/data_eval --data_dir=../../Data/ADEChallenge/data_train --subset=validation --checkpoint_dir=./train/Nov11/Down32/ADE_16_8_3_noAttWeight_shareWeight --run_once --out_label_dir=./results/Nov11/Down32/ADEChallenge_16_8_3_noAttWeight_shareWeight_sumDAG_12000 --num_examples=2000 --image_size=256 --batch_size=32 --reduce_size=32 --block_size=8 --hid_size=16 --use_attention_weight=False --database=ADEChallenge


CUDA_VISIBLE_DEVICES=0 python DAGResnet_multi_gpu_train.py --data_dir=../../Data/SiftFlow_Oct30_new/data_train --batch_size=32 --train_dir=./train/Nov11/Down32/SiftFlow_16_8_3_noAttWeight_shareWeight_newRelu_lr0.01 --learning_rate=0.01 --image_size=256 --use_attention_weight=False --reduce_size=32 --block_size=8 --hid_size=16 --usePredictionMax=False --database=SiftFlow --num_gpus=1 --do_val=500
CAMVID
EVAL V0.6.5
--eval_dir=../../Data/CamVid/data_eval --data_dir=../../Data/CamVid/data_train --subset=validation --checkpoint_dir=./train/Nov11/Down32/CamVid_8_8_3_noAttWeight_shareWeight_lr_0.01_newRelu --run_once --out_label_dir=./results/Nov11/Down32/CamVid_8_8_3_noAttWeight_shareWeight_lr_0.01_newRelu_14000 --num_examples=1500 --image_size=256 --batch_size=32 --reduce_size=32 --block_size=8 --hid_size=8 --use_attention_weight=False --database=CamVid


CUDA_VISIBLE_DEVICES=0 python --eval_dir=../../Data/SiftFlow_Oct30_new/data_eval --data_dir=../../Data/SiftFlow_Oct30_new/data_train --subset=validation --checkpoint_dir=./train/Nov11/Down32/SiftFlow_16_8_3_shareWeight_noAttWeight_finetune_lr0.001 --run_once --out_label_dir=./results/Nov11/Down32/SiftFlow_16_8_3_shareWeight_noAttWeight_finetune_lr0.001_27000 --num_examples=2000 --image_size=256 --batch_size=32 --reduce_size=32 --block_size=8 --hid_size=8 --use_attention_weight=False --database=SiftFlow


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python DAGResnet_multi_gpu_train.py --data_dir=../../Data/SiftFlow_Oct30_new/data_train --batch_size=192 --train_dir=./train/Nov11/Down32/SiftFlow_16_8_3_shareWeight_noAttWeight_finetune_lr0.001 --learning_rate=0.001 --image_size=256 --use_attention_weight=False --reduce_size=32 --block_size=8 --hid_size=16 --usePredictionMax=False --database=SiftFlow --num_gpus=6 --do_val=500


ADE Data
--train_directory=/media/Babylon/dcnhan/Code/SceneParsing/Data/ADEChallenge/train/images --train_label_directory=/media/chinhan/1CF51B3B19F30D68/Databases/ADEChallenge/ADEChallengeData2016/siftflow_class_label/training --validation_directory=/media/Babylon/dcnhan/Code/SceneParsing/Data/ADEChallenge/validation/images --val_label_directory=/media/chinhan/1CF51B3B19F30D68/Databases/ADEChallenge/ADEChallengeData2016/siftflow_class_label/validation --output_directory=/media/Babylon/dcnhan/Code/SceneParsing/Data/ADEChallenge/data_train_new
