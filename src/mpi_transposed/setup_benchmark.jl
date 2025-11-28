using LinearAlgebra
using SparseArrays
using Random
using Base.Threads
using InteractiveUtils

function threaded_fill!(X::Matrix{Float64}, value::Float64)
    @threads for j in 1:size(X, 2)
        @simd for i in 1:size(X,1)
            @inbounds X[i,j] = value
        end
    end
end


# --- 1. Configuration (EPYC scale) ---
# WE set N = M for a square dense matrix
const N = 50_000
const M = 50_000
const DENSITY = 15.0/N # density of the sparse matrix, ~15 nonzeros per row
const GB_PER_MATRIX = (N * M * 8) / (1024^3) # size of one dense matrix in GB

println("--- DEUCALION BADNWIDTH TEST ---")
versioninfo()
println(" --- SETUP PHASE ---")
println("Available Threads: ", nthreads())
println("Matrix: $N x $M")
println("Size: $(round(GB_PER_MATRIX, digits=2)) GB per dense matrix")


# --- 1. Compiler Warmup ---
println("\n[1/3] Compiler Warmup...")
X_dummy = Matrix{Float64}(undef, 1024, 1024)
threaded_fill!(X_dummy, 0.0)
X_dummy = nothing #discard
GC.gc()
println(" Warmup complete.")





#= 
Logical: (N X M) -> 40k x 40k
Physical: (M X N) -> 40k x 40k (transposed for better memory access)
=#

println("\n[2/3] Allocating dense matrices in transposed format...")

t_alloc = @elapsed begin
    X_phys = Matrix{Float64}(undef, M, N)
end



println("\n[3/3] Benchmark Initialization of dense matrix...")
GC.gc()

time_write = @elapsed begin
    threaded_fill!(X_phys, 0.0)
end

# 4. Benchmark (Rewrite - Steady State)
# We run it AGAIN to see pure bandwidth without OS Page Fault overhead
println("      Running Rewrite (Steady State)...")
t_rewrite = @elapsed begin
    threaded_fill!(X_phys, 2.0)
end

bw_first = GB_PER_MATRIX / time_write
bw_steady = GB_PER_MATRIX / t_rewrite

println("\n--- RESULTS ---")
println("First Touch Time: $(round(time_write, digits=4)) seconds")
println("First Touch BW: $(round(bw, digits=2)) GB/s")
println("---")
println("Steady State Rewrite Time: $(round(t_rewrite, digits=4)) seconds")
println("Steady State Rewrite BW: $(round(bw_steady, digits=2)) GB/s")

# --- 4. Verification ---
if bw_steady > 250
    println("\n SUCESS: Excellent NUMA saturation detected! ")
elseif bw_steady > 150
    println("\n SUCCESS: Good NUMA saturation detected. ")
else
    println("\n FAILURE: Poor NUMA saturation detected! Please check your system configuration. ")
end