gcloud ml-engine local train --module-name $MAIN_TRAINER_MODULE \
    --job-dir 'output' \
    --verbosity debug \
    -- \
    --traindir $TRAINDIR \
    --evaldir $EVALDIR \
    --bucket $BUCKET \
    --outputdir 'output' \
    --dropout 0.27851938277724281 \
    --batchsize 484 \
    --epochs 100 \
    --hidden_units '120, 64,12' \
    --feat_eng_cols 'ON' \
    --learn_rate 0.082744837373068689
    