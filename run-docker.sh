
docker run -it --gpus all \
       -v /home/ubuntu/virginia/home/jgibson/data/hypersim/:/data/hypersim:ro \
       -v /home/ubuntu/scratch/:/scratch/ \
       --ipc=host \
       --net=host \
       ubuntu-romav2
