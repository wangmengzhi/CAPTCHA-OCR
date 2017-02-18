
set -e
config=vgg_16_captcha.py
output=./captcha_vgg_model

paddle train \
--config=$config \
--use_gpu=0 \
--trainer_count=8 \
--num_passes=1 \
--save_dir=$output \
--init_model_path=./captcha_vgg_model/pass-00000 \
--config_args=trainone=True \
