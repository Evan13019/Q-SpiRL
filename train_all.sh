#!/bin/bash
#SBATCH --job-name=dsqn_combo
#SBATCH --array=0-7
#SBATCH --time=15:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

# ---- conda ----
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate qml


TASK_ID=${SLURM_ARRAY_TASK_ID}
# ---- decode configuration ----
if [ "${TASK_ID}" -lt 2 ]; then
    # quantum = 0 (qubits irrelevant)
    quantum=0
    qmlp=${TASK_ID}     # 0 or 1
    qubits=1            # dummy value, ignored by code

else
    # quantum = 1
    quantum=1

    # shift index
    idx=$((TASK_ID - 2))

    QMLP_LIST=(0 1)
    QUBITS_LIST=(2 4 8)

    qmlp_idx=$(( idx / 3 ))
    qubits_idx=$(( idx % 3 ))

    qmlp=${QMLP_LIST[$qmlp_idx]}
    qubits=${QUBITS_LIST[$qubits_idx]}
fi

echo "TASK ${TASK_ID}: qmlp=${qmlp}, quantum=${quantum}, qubits=${qubits}"
dir_idx="5"
mkdir -p "models${dir_idx}"
cp train.py "models${dir_idx}/train.py.txt"
cp env.py   "models${dir_idx}/env.py.txt"
cp net.py   "models${dir_idx}/net.py.txt"
cp Qnet.py   "models${dir_idx}/Qnet.py.txt"

# ---- flags ----
qmlp_flag=""
quantum_flag=""

if [ "${qmlp}" -eq 1 ]; then
    qmlp_flag="--qmlp"
fi
if [ "${quantum}" -eq 1 ]; then
    quantum_flag="--quantum"
fi

# ---- run ----
python -u train.py \
    ${qmlp_flag} \
    ${quantum_flag} \
    --qubits "${qubits}" \
    --episodes 800 \
    --model_dir "models${dir_idx}" \
    --plot_root "plots${dir_idx}-env" \
    --plots \
    --device cpu
