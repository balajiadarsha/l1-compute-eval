
################################
# Ray Management Functions     #
################################

# Function to stop Ray
stop_ray() {
    echo "Stopping Ray on $(hostname)"
    ray stop -f
}


# Function to start Ray head node
start_ray_head() {
    set -x  # Enable command echo for debugging

    current_node=$(hostname)
    ray_port=6380

    echo "Starting Ray head on $current_node"
    # Start Ray head node
    ray start --num-cpus=64 --num-gpus=8 --head --port=$ray_port
    # Wait for Ray head to be up
    echo "Waiting for Ray head to be up..."
    until ray status &>/dev/null; do
        sleep 5
        echo "Waiting for Ray head..."
    done
    echo "ray status: $(ray status)"
    echo "Ray head node is up."
}

# Function to start Ray worker node
start_ray_worker() {
    set -x  # Enable command echo for debugging

    current_node=$(hostname)
    ray_port=6380

    # Debug: Echo OMP_NUM_THREADS and hostname
    echo "OMP_NUM_THREADS on $current_node: $OMP_NUM_THREADS"

    echo "Starting Ray worker on $current_node, connecting to $RAY_HEAD_IP:$ray_port"
    # Start Ray worker node
    ray start --num-gpus=1 --num-cpus=1 --address=$RAY_HEAD_IP:$ray_port

    # Wait for Ray worker to be up
    echo "Waiting for Ray worker to be up..."
    until ray status &>/dev/null; do
        sleep 5
        echo "Waiting for Ray worker..."
    done

    echo "ray status: $(ray status)"
    echo "Ray worker node is up."
}

# Function to verify Ray cluster status
verify_ray_cluster() {
    local expected_nodes="$1"
    local attempts=0
    local max_retries=2

    while [ $attempts -le $max_retries ]; do
        echo "Verifying Ray cluster status (Attempt $((attempts + 1)))..."

        # Extract active nodes count from ray status
        actual_nodes=$(ray status | awk '
            /^Node status/ { in_active=0 }
            /^Active:/ { in_active=1; next }
            /^Pending:/ { in_active=0 }
            in_active && /^ *[0-9]+ node_/ { count++ }
            END { print count }
        ')

        if [ "$actual_nodes" -eq "$expected_nodes" ]; then
            echo "Ray cluster has the expected number of nodes: $actual_nodes"
            return 0
        else
            echo "Ray cluster has $actual_nodes active nodes, expected $expected_nodes."
            if [ $attempts -lt $max_retries ]; then
                echo "Attempting to restart Ray cluster..."

                # Stop Ray on all nodes
                echo "Stopping Ray on all nodes..."
                mpiexec -n "$expected_nodes" -hostfile "$PBS_NODEFILE" bash -c "source $COMMON_SETUP_SCRIPT; setup_environment; stop_ray"

                # Start Ray head node
                echo "Starting Ray head node..."
                mpiexec -n 1 -host "$head_node" bash -l -c "source $COMMON_SETUP_SCRIPT; setup_environment; start_ray_head"

                # Start Ray worker nodes
                echo "Starting Ray worker nodes..."
                for worker in "${worker_nodes[@]}"; do
                    echo "Starting Ray worker on $worker"
                    mpiexec -n 1 -host "$worker" bash -l -c "source $COMMON_SETUP_SCRIPT; setup_environment; start_ray_worker"
                done

                # Allow some time for Ray to initialize
                sleep 10
            fi
        fi
        attempts=$((attempts + 1))
    done

    echo "Ray cluster verification failed after $max_retries retries. Exiting."
    exit 1
}


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
export RAY_ADDRESS="$RAY_HEAD_IP:6380"

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
mpiexec -n "$num_nodes" -hostfile "$PBS_NODEFILE" bash -c "source $COMMON_SETUP_SCRIPT; setup_environment; stop_ray; cleanup_python_processes;"

# Start Ray head node
echo "Starting Ray head node..."
mpiexec -n 1 -host "$head_node" bash -l -c "source $COMMON_SETUP_SCRIPT; export RAY_HEAD_IP=$RAY_HEAD_IP; setup_environment; start_ray_head"

echo "Starting Ray worker nodes..."
for worker in "${worker_nodes[@]}"; do
    echo "Starting Ray worker on $worker"
    mpiexec -n 1 -host "$worker" bash -l -c "source $COMMON_SETUP_SCRIPT; export RAY_HEAD_IP=$RAY_HEAD_IP; setup_environment; start_ray_worker"
done

# Verify Ray cluster status
echo "Verifying Ray cluster status..."
verify_ray_cluster "$num_nodes"

echo "Ray cluster setup complete."
