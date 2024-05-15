python train_agent.py \
    --env "ToyMC-v0" \
    --log-interval 10 \
    --save-interval 10 \
    --procs 1 \
    --frames-per-proc 1000 \
    --discount 0.94 \
    --epochs 4 \
    --lr 0.0003 \
    --eval \
    --eval-episodes "-1" \
    --frames "-1" \
    "$@"
