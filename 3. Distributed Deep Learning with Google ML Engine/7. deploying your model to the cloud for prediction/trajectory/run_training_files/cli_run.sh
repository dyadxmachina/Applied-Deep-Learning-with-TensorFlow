gcloud ml-engine jobs submit training $JOB_NAME \
        --stream-logs \
        --package-path $TRAINER_PACKAGE_PATH \
        --module-name $MAIN_TRAINER_MODULE \
        --job-dir $JOB_DIR \
        --runtime-version $RUNTIME_VERSION \
        --region $REGION \
        --config $CONFIG \
        -- \
        --traindir $TRAINDIR \
        --evaldir $EVALDIR \
        --bucket $BUCKET \
        --outputdir $JOB_DIR \
        --dropout 0.27851938277724281 \
        --batchsize 4 \
        --epochs 100 \
        --hidden_units '64,24,12' \
        --feat_eng_cols 'ON' \
        --learn_rate 0.082744837373068689

