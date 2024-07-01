pip install -r requirements.txt
pip install -r requirements_after.txt
#pip install -U mmcv
#mim install mmcv
if [ "$(uname -s)" == "Linux" ]; then
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html
else
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7.0/index.html
fi



