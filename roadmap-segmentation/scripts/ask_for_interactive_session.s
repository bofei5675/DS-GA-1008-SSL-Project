# request a gpu
srun -t3:30:00 --mem=16000 --gres=gpu:p40:1 --pty /bin/bash

# request 2 cpu
srun -c1 -t2:00:00 --mem=16000 --pty /bin/bash
