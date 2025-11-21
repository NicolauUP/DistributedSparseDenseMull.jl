#!/bin/bash

# --- Configurações de Trabalho ---
export BENCH_N=50000
export ARCH_NAME="AMD_EPYC_7742_MULTINODE"
CPUS_PER_NODE=32 # O limite de Workers estável que definiste
TIME_LIMIT="02:00:00"

# --- Ficheiros ---
RESULTS_FILE="../results/architecture_benchmark_results_multinode.csv"
LOGS_DIR="../logs/multinode"
mkdir -p "$LOGS_DIR"

# --- Criação do CSV Header (Apenas se não existir) ---
# Adicionamos a coluna 'NODES' para análise multi-nodo
# Create CsV Header if it does not exist
if [ ! -f "../results/architecture_benchmark_results.csv" ]; then
    echo "Architecture,N,Workers,Nodes,Threads,Time(s),GFlops,Time Setup H(s),Time Setup X(s),Time Gather(s),Min Compute Time(s),Max Compute Time(s),Pct Imbalance (%)" > ../results/architecture_benchmark_results_multinode.csv
fi

# --- Loop de Testes Multi-Nodo (2 a 8 Nós) ---
for NODES in {2..8}; do
    # Calcular o total de Workers de Computação
    WORKERS=$((NODES * CPUS_PER_NODE))
    TASKS=$((WORKERS + 1)) # +1 para o processo Master

    echo "Submitting Multi-Node benchmark with $NODES nodes, $WORKERS workers ($TASKS tasks total)"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=arch_bench_N${NODES}_W${WORKERS}
#SBATCH --output=${LOGS_DIR}/bench_N${NODES}_W${WORKERS}.out
#SBATCH --nodes=${NODES}                  # <<< NÓS VARIÁVEIS (2 a 8)
#SBATCH --ntasks=${TASKS}                 # Total de Workers + Master
#SBATCH --cpus-per-task=1                 # PURE MPI (Estratégia mais estável)
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=normal-x86
#SBATCH -A f202409396cpcaa2x   
#SBATCH --mem=128G

# Variável de ambiente para registar o número de nós
export BENCH_NODES=${NODES}



julia --project=../. ../src/bench_distributed_run.jl
EOT

done

echo "Todos os jobs multi-nodo foram submetidos."