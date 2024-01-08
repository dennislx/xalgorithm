export CUDA_VISIBLE_DEVICES=1,2,3 &&
debugpy-run \
    -p :5561 \
    mentallama.py
