# python train.py --fold 0,1,2 \
#                 --kernel-type test-name3 \
#                 --tags performance,test \
#                 --data-dir ./data/ \
#                 --data-folder 512 \
#                 --image-size 384 \
#                 --enet-type efficientnet_b3 \
#                 --use-warmup \
#                 --init-lr 3e-5 \
#                 --use-memory-profiler

python train.py --fold 0 \
                --kernel-type test-name3 \
                --tags performance,test \
                --data-dir ./data/ \
                --data-folder 512 \
                --image-size 384 \
                --enet-type efficientnet_b3 \
                --use-warmup \
                --init-lr 3e-5 \
                --use-profiler \
                --epoch 5 \
                --DEBUG