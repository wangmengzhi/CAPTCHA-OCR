
set -e
config=vgg_16_captcha.py
output=./captcha_vgg_model
log=train.log

paddle train \
--config=$config \
--use_gpu=0 \
--trainer_count=8 \
--num_passes=1 \
--save_dir=$output \
2>&1 | tee $log

python -m paddle.utils.plotcurve -i $log > plot.png

