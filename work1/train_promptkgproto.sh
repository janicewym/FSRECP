CUDA_VISIBLE_DEVICES=0,1 \
python -u train_promptkgproto.py \
        --train_iter 20000 --val_step 1000 --val_iter 1000 \
        --trainN 5 --N 5 --K 1 --Q 1 \
        --hidden_size 768 \
        --batch_size 4 \
        --prompt \
        --kg \
        --type \
        --encoder "kgtypeprompt" \
        --model "my_kgproto" \
        --val "val_wiki" \
        --test "test_wiki"\
        --root "../../KGPrompt_v4_1/data/" \
        --version "kgproto_v1" 