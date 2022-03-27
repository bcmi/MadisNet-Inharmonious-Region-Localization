MODEL=dirl

#NAME=madisnet_triW_${LAMBDA_TRI}_iou_${LAMBDA_IOU}_m${M}_${MODEL}


python3  -u test.py \
--dataset_root /media/sda/datasets/IHD \
--checkpoints_dir /media/sda/Harmonization/MadisNet/${MODEL} \
--batch_size 1 \
--gpu_ids 0 \
--preprocess resize \
--no_flip \
--save_epoch_freq 5 \
--is_train 0 \
--batch_norm \
--model ${MODEL} \
--resume -2 \
# > ${NAME}best.log 2>&1 & 


