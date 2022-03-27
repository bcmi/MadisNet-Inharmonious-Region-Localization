LAMBDA_ATTENTION=1
LAMBDA_TRI=0.001
LAMBDA_REG=0.001
LAMBDA_SSIM=0
LAMBDA_IOU=1
M=0.001
MODEL=dirl #unet

NAME=madisnet_triW_${LAMBDA_TRI}_iou_${LAMBDA_IOU}_m${M}_${MODEL}


python3 -u train.py \
--dataset_root /media/sda/datasets/IHD \
--checkpoints_dir /media/sda/Harmonization/inharmonious/DIRLNet/${NAME} \
--batch_size 12 \
--gpu_ids 0 \
--preprocess resize_and_crop \
--save_epoch_freq 5 \
--is_train 1 \
--lr 1e-4 \
--nepochs 60 \
--lambda_attention ${LAMBDA_ATTENTION} \
--lambda_ssim ${LAMBDA_SSIM} \
--lambda_iou ${LAMBDA_IOU} \
--lambda_tri ${LAMBDA_TRI} \
--lambda_reg ${LAMBDA_REG} \
--batch_norm \
--model ${MODEL} \
--m ${M} \
> ${NAME}.log 2>&1 & 
