gcloud ml-engine jobs submit training $JOB_NAME \
        --package-path $TRAINER_PACKAGE_PATH \
        --module-name $MAIN_TRAINER_MODULE \
        --job-dir $JOB_DIR \
        --runtime-version $RUNTIME_VERSION \
        --region $REGION \
        --config config.yaml \
        --verbosity debug \
        -- \
        --traindir $TRAINDIR \
        --evaldir $EVALDIR \
        --bucket $BUCKET \
        --outputdir $OUTPUTDIR \
        --batchsize 32 \
        --epochs 10 

        