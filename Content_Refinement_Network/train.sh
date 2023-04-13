python train.py --warmup \
                --win_size 10 \
                --train_ps 320 \
                --gpu 0 \
                --datasets_dir /mnt/188/b/NTIRE23/data/shadow/ISTD/ISTD_Dataset/train \
                --train_list /mnt/188/b/NTIRE23/data/shadow/ISTD/ISTD_Dataset/train/train_list.txt \
                --valset_dir /mnt/188/b/NTIRE23/data/shadow/ISTD/ISTD_Dataset/test \
                --validation_list /mnt/188/b/NTIRE23/data/shadow/ISTD/ISTD_Dataset/test/val_list.txt \
                --batch_size 3
