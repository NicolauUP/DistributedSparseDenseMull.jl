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
const N = 20_000
const M = 20_000
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

bw = GB_PER_MATRIX / time_write

println("\n--- RESULTS ---")
println("Time: $(round(time_write, digits=4)) seconds")
println("Bandwidth: $(round(bw, digits=2)) GB/s")

# --- 4. Verification ---
if bw > 250
    println("\n SUCESS: Excellent NUMA saturation detected! ")
elseif bw > 150
    println("\n SUCCESS: Good NUMA saturation detected. ")
elseif bw > 100
    println("\n WARNING: Moderate NUMA saturation detected. Consider tuning your environment. ")
else
    println("\n FAILURE: Poor NUMA saturation detected! Please check your system configuration. ")
end