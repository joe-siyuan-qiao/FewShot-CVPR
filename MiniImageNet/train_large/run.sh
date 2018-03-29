th main.lua \
    -netType resnet \
    -simpleconvnet true \
    -dataset miniimagenet \
    -batchSize 64 \
    -data ../data/ \
    -nEpochs 300 > training.log

mkdir -p ../few_shot/pretrained
cp checkpoints/model_best.t7 ../few_shot/pretrained/model.t7
