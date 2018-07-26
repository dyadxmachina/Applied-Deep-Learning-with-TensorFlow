export PYTHONPATH=${PYTHONPATH}:${PWD}/trajectory

python -m train.task --batchsize=50 --traindir=../../data/train --evaldir=../../data/test --epochs=1 --outputdir=output