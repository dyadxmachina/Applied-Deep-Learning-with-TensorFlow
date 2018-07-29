export PYTHONPATH=${PYTHONPATH}:${PWD}/trajectory

python -m train.task --batchsize=50 --traindir=data/train --evaldir=data/test --epochs=1 --outputdir=8 --feat_eng_cols=1 --job-dir='none' --hidden_units='64,12'

gcloud ml-engine jobs describe $JOB_NAME