CUDA_VISIBLE_DEVICES=2,3 \
python -u train_con.py \
        --train_iter 20000 --val_step 1000 --val_iter 1000 \
        --trainN 5 --N 5 --K 1 --Q 1 \
        --hidden_size 768 \
        --batch_size 4 \
        --prompt \
        --kg \
        --type \
        --con \
        --encoder "kgtypeprompt" \
        --model "my_kgcon" \
        --val "val_wiki" \
        --test "test_wiki"\
        --root "../../KGPrompt_v4_1/data/" \
        --version "con_v1" 