CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 \
python -u train_two_con_2.py \
        --train_iter 20000 --val_step 1000 --val_iter 1000 \
        --encoder kgtypeprompt --model proto \
        --trainN 10 --N 10 --K 5 --Q 1 \
        --hidden_size 768 \
        --batch_size 4 \
        --prompt \
        --val val_wiki --test test_wiki \
        --version "v2-exp5-twocon-sub-v2" \
        --kg \
        --type \
        --con \
        --fp16 
#        --load_ckpt "checkpoint/proto-v-exp5-twocon-1w-sub-kgtypeprompt-train_wiki-val_wiki-10-5.pth.tar"