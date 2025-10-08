#!/bin/bash -l
#PBS -N safety_rs
#PBS -l select=1:ncpus=32
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -A argonne_tpc
#PBS -l filesystems=home:eagle

source ~/.bashrc

conda activate /lus/eagle/projects/argonne_tpc/abalaji/conda_env/reasoning 

# Read nodes from PBS_NODEFILE
nodes=($(sort -u "$PBS_NODEFILE"))
num_nodes=${#nodes[@]}

# Get the current node's hostname (assumed to be the head node)
head_node=$(hostname | sed 's/.lab.alcf.anl.gov//')

echo "Nodes: ${nodes[@]}"
echo "Head node: $head_node"

# Get the IP address of the head node
RAY_HEAD_IP=$(getent hosts "$head_node" | awk '{ print $1 }')
echo "Ray head IP: $RAY_HEAD_IP"

# Export variables for use in functions
export head_node
export RAY_HEAD_IP
export HOST_IP="$RAY_HEAD_IP"
export RAY_ADDRESS="$RAY_HEAD_IP:6379"

# Define worker nodes (exclude head node)
worker_nodes=()
for node in "${nodes[@]}"; do
    short_node=$(echo "$node" | sed 's/.lab.alcf.anl.gov//')
    if [ "$short_node" != "$head_node" ]; then
        worker_nodes+=("$short_node")
    fi
done

echo "Worker nodes: ${worker_nodes[@]}"

# Stop Ray on all nodes using mpiexec
echo "Stopping any existing Ray processes on all nodes..."
mpiexec -n "$num_nodes" -hostfile "$PBS_NODEFILE" bash -c "source ~/.bashrc; conda activate /lus/eagle/projects/argonne_tpc/abalaji/conda_env/reasoning; stop_ray; cleanup_python_processes;"

# Start Ray head node
echo "Starting Ray head node..."
mpiexec -n 1 -host "$head_node" bash -l -c "source ~/.bashrc; conda activate /lus/eagle/projects/argonne_tpc/abalaji/conda_env/reasoning; export RAY_HEAD_IP=$RAY_HEAD_IP; start_ray_head"

echo "Starting Ray worker nodes..."
for worker in "${worker_nodes[@]}"; do
    echo "Starting Ray worker on $worker"
    mpiexec -n 1 -host "$worker" bash -l -c "source ~/.bashrc; conda activate /lus/eagle/projects/argonne_tpc/abalaji/conda_env/reasoning;  export RAY_HEAD_IP=$RAY_HEAD_IP; setup_environment; start_ray_worker"
done

# Verify Ray cluster status
echo "Verifying Ray cluster status..."
verify_ray_cluster "$num_nodes"

echo "Ray cluster setup complete."
