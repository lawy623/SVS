#!/usr/bin/env sh
set -e

/usr/local/MATLAB/R2015a/bin/matlab -nodisplay -nosplash -nodesktop -r train_svs 2>&1|tee ./prototxt/svs/train.log

