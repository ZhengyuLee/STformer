export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 1 --num_epochs 10   --batch_size 96  --mode train --dataset SMD --data_path dataset/SMD        --enc_in 38  --output_c 38
python main.py --anormly_ratio 1  --num_epochs 10       --batch_size 96     --mode test    --dataset SMD --data_path dataset/SMD   --enc_in 38  --output_c 38

python main.py --anormly_ratio 1 --num_epochs 3    --batch_size 128  --mode train --dataset PSM --data_path dataset/PSM        --enc_in 25  --output_c 25
python main.py --anormly_ratio 1  --num_epochs 10       --batch_size 128     --mode test    --dataset PSM --data_path dataset/PSM   --enc_in 25  --output_c 25

python main.py --anormly_ratio 1 --num_epochs 3    --batch_size 64  --mode train --dataset MSL --data_path dataset/MSL       --enc_in 55  --output_c 55
python main.py --anormly_ratio 1  --num_epochs 10       --batch_size 64     --mode test    --dataset MSL --data_path dataset/MSL   --enc_in 55  --output_c 55

python main.py --anormly_ratio 1 --num_epochs 3    --batch_size 80  --mode train --dataset SMAP --data_path dataset/SMAP        --enc_in 25  --output_c 25
python main.py --anormly_ratio 1  --num_epochs 10       --batch_size 80    --mode test    --dataset SMAP --data_path dataset/SMAP   --enc_in 25  --output_c 25

python main.py --anormly_ratio 0.3 --num_epochs 3   --batch_size 64  --mode train --dataset SMD  --data_path dataset/SMD   --input_c 38
python main.py --anormly_ratio 0.3 --num_epochs 10   --batch_size 64     --mode test    --dataset SMD   --data_path dataset/SMD     --input_c 38     --pretrained_model 20