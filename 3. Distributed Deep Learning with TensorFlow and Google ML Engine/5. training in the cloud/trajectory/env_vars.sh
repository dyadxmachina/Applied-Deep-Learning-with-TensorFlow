
export TRAINER_PACKAGE_PATH="train"
now=$(date +"%Y%m%d_%H%M%S")
export JOB_NAME="trajectory_$now"
export MAIN_TRAINER_MODULE="train.task"
export JOB_DIR="gs://trajectory/output"
export PACKAGE_STAGING_LOCATION="gs://trajectory/model"
export REGION="us-central1"
export RUNTIME_VERSION="1.6"
export TRAINDIR="gs://trajectory/data/train"
export EVALDIR="gs://trajectory/data/test"
export OUTPUTDIR="gs://trajectory/output"
export BUCKET="trajectory"

