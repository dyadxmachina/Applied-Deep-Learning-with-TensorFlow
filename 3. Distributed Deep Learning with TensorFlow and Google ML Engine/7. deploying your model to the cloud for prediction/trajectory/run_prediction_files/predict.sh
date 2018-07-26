
gcloud ml-engine models list
gcloud ml-engine models create trajectory --regions us-central1 --enable-logging
gcloud ml-engine models list
gcloud ml-engine versions list --model trajectory
gcloud ml-engine versions create v2 --model trajectory2 --origin $MODEL_BINARY --runtime-version 1.6
gcloud ml-engine versions create v2 --model trajectory2 --origin $MODEL_BINARY --runtime-version 1.6 --config config_predictions.yaml
gcloud ml-engine predict --model trajectory2 --version v1 --json-instances batch_predict.json