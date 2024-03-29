!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=20   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=100000M        # memory per node
#SBATCH --time=0-2:30      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

source tensorflow/bin/activate 
module load scipy-stack
pip install tensorflow==2.9.0
pip install sklearn
pip install shap
module load cuda cudnn 
export TF_GPU_ALLOCATOR=cuda_malloc_async
python n_clients_improoved_FL.py
