using Distributed
using LinearAlgebra
using SparseArrays
using Printf
using ClusterManagers


#Detect Slurm environment and add workers accordingly
if "SLURM_NTASKS" in keys(ENV)
    total_tasks = parse(Int, ENV["SLURM_NTASKS"])

    #On a single node, you can use addprocs locally

    n_workers = max(1, total_tasks - 1) # Reserve 1 for master

    println("Detected SLURM environment with $total_tasks tasks. Adding $n_workers local workers.")


    addprocs(SlurmManager(n_workers))
else
    println("Master: No SLURM environment detected. Adding 4 local workers for testing.")
    addprocs(4)
end




println("Cluster Active: 1 Master + $(nworkers()) workers.")

@everywhere begin
    using LinearAlgebra
    using SparseArrays

    include("distributed_worker.jl")
end

# --- B Create Hamiltonian Matrix H --- #

N = 50000

println("Master: Generating sparse Hamiltonian matrix H of size $(N)x$(N)...")

H_master = sprand(N, N, 0.01)
H_master = H_master + H_master'  # Make it symmetric

println("Master: Distributing H matrix to workers...")


@everywhere procs() worker_init_H($H_master)

println("Master: H matrix distributed to all workers.")

# --- Distribute Slabs ---
println("\nMaster: Distributing slabs to workers...")

workers_list = workers()
num_workers = length(workers_list)
slab_size = div(N, num_workers)

for (i,pid) in enumerate(workers_list)

    start_idx = 1 + (i - 1) * slab_size
    #Ensure the last worker gets any remaining rows
    end_idx = (i == num_workers) ? N : i * slab_size

    col_range = start_idx:end_idx 


    remotecall_wait(worker_allocate_slabs, pid, N, col_range)
end 

println("Master: Slabs distributed to all workers.")

# --- Run Calculation

println("\nMaster: Starting distributed matrix-vector multiplication...")

t_start = time()
futures = []

for pid in workers_list
    f = remotecall(worker_matmul, pid)
    push!(futures, f)
end

println("Master: Waiting for results from workers...")

#Barrier - wait for all workers to finish
wait.(futures)

t_end = time()
println("Master: Distributed matrix-vector multiplication completed in $(round(t_end - t_start, digits=2)) seconds.")

println("\nMaster: Gathering results from workers...")

# Diagonals from the workes, for example!
diag_futures = [remotecall(worker_get_diagonal, pid) for pid in workers_list]
diagonal_results = fetch.(diag_futures)

full_diagonal = reduce(vcat, diagonal_results)

println("Master: Gathered full diagonal of result matrix with length $(length(full_diagonal)).")
println("Master: Distributed computation finished successfully.")

# --- Clean Up Workers --- #
println("\nMaster: Cleaning up worker storage...")
for pid in workers_list
    remotecall_wait(worker_clear_storage, pid)
end
println("Master: Worker storage cleared.")
rmprocs(workers_list)
println("Master: All workers removed. Program complete.")