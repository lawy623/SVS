#!/usr/bin/env sh
set -e

/usr/local/MATLAB/R2015a/bin/matlab -nodisplay -nosplash -nodesktop -r train_viewSyn 2>&1|tee ./prototxt/viewSynthesis_BN/train.log

