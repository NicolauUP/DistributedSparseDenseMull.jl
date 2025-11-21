using Distributed
using LinearAlgebra
using SparseArrays
using Printf
using SlurmClusterManager


# --- Configuration ---
N = parse(Int, get(ENV, "BENCH_N", "50000")) # Size of the Hamiltonian matrix
threads_per_worker = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))



if "SLURM_JOB_ID" in keys(ENV)
    println("Running inside a SLURM job. Initializing SlurmClusterManager...")
    addprocs(SlurmManager())
else
    println("Not running inside a SLURM job. Adding local workers...")
    addprocs(4)
end



n_workers = nworkers()
println("BENCHMARK : N = $N, Workers = $n_workers, Threads per worker = $threads_per_worker")

@everywhere worker() begin
    ENV["JULIA_WORKER_THREADS"] = $threads_per_worker
end

@everywhere begin
    using LinearAlgebra
    using SparseArrays
    include(joinpath(@__DIR__,"distributed_worker.jl"))
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

# --- Warmup ---
println("\nMaster: Starting warmup matrix-vector multiplication...")

wait([remotecall(worker_matmul, pid) for pid in workers_list])

# -- - Benchmark Distributed Matrix-Vector Multiplication --- #


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

time = t_end - t_start
@printf("Distributed MatMul Time: %.4f seconds\n", time)
gflops = (2.0 * N^2) / (time * 1e9)
@printf("Performance: %.2f GFLOPS\n", gflops)

# ---  Save Results --- #
arch = get(ENV, "ARCH_NAME", "Unknown")
result_line = "$arch, $N, $n_workers, $threads_per_worker, $(round(time, digits=4)), $(round(gflops, digits=2))\n"

println("\nFinal Results: $result_line")

open("distributed_benchmark_results.csv", "a") do io
    write(io, result_line)
end

# --- Clean Up Workers --- #
println("\nMaster: Cleaning up worker storage...")
for pid in workers_list
    remotecall_wait(worker_clear_storage, pid)
end
println("Master: Worker storage cleared.")
rmprocs(workers_list)
println("Master: All workers removed. Program complete.")