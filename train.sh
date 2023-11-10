python train.py --kernel-type run01 --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --n-epochs 18 --use-warmup --init-lr 3e-5 

python train.py --kernel-type run02 --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type efficientnet_b3 --n-epochs 18 --use-warmup --init-lr 3e-5 

python train.py --kernel-type run03 --data-dir ./data/ --data-folder 768 --image-size 768 --enet-type efficientnet_b3 --n-epochs 18 --use-warmup --init-lr 3e-5 

python train.py --kernel-type run04 --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type tf_efficientnet_b5_ns --n-epochs 18 --use-warmup --init-lr 3e-5 

python train.py --kernel-type run05 --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type tf_efficientnet_b7_ns --n-epochs 18 --use-warmup --init-lr 3e-5 

python train.py --kernel-type run06 --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type resnest101 --n-epochs 18 --use-warmup --init-lr 3e-5 

python train.py --kernel-type run07 --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type seresnext101 --n-epochs 18 --use-warmup --init-lr 3e-5 

python train.py --kernel-type run08 --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type seresnext101 --n-epochs 18 --init-lr 3e-5 

python train.py --kernel-type run09 --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --n-epochs 18 --use-warmup --init-lr 2e-5 

python train.py --kernel-type run10 --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --n-epochs 18 --use-warmup --init-lr 1.5e-5 

python train.py --kernel-type run11 --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --n-epochs 18 --use-warmup --init-lr 3e-5 --use-meta

python train.py --kernel-type run12 --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --n-epochs 18 --use-warmup --init-lr 3e-5 --num-dropout 1

python train.py --kernel-type run13 --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --n-epochs 18 --use-warmup --init-lr 3e-5 --num-dropout 3

#python train.py --kernel-type 9c_b4ns_2e_896_ext_15ep --data-dir ./data/ --data-folder 1024 --image-size 896 --enet-type tf_efficientnet_b4_ns --init-lr 2e-5 
#
#python train.py --kernel-type 9c_b4ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b4_ns 
#python train.py --kernel-type 9c_b4ns_768_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns 
#python train.py --kernel-type 9c_b4ns_768_768_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 768 --enet-type tf_efficientnet_b4_ns 
#python train.py --kernel-type 9c_meta_b4ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns --use-meta 
#python train.py --kernel-type 4c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns --out-dim 4 --init-lr 1.5e-5 
#python train.py --kernel-type 9c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns --init-lr 1.5e-5 
#python train.py --kernel-type 9c_b5ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b5_ns 
#python train.py --kernel-type 9c_meta128_32_b5ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b5_ns --use-meta --n-meta-dim 128,32 
#python train.py --kernel-type 9c_b6ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b6_ns 
#python train.py --kernel-type 9c_b6ns_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b6_ns 
#python train.py --kernel-type 9c_b6ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b6_ns 
#python train.py --kernel-type 9c_b7ns_1e_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b7_ns --init-lr 1e-5 
#
#python train.py --kernel-type 9c_b7ns_1e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b7_ns --init-lr 1e-5 
#
#python train.py --kernel-type 9c_meta_1.5e-5_b7ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b7_ns --use-meta 
#python train.py --kernel-type 9c_nest101_2e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type resnest101 --init-lr 2e-5 
#python train.py --kernel-type 9c_se_x101_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type seresnext101 
