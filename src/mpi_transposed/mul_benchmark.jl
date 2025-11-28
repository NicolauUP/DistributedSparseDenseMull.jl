using LinearAlgebra
using SparseArrays
using Random
using Base.Threads
using InteractiveUtils
using ThreadPinning
using LoopVectorization

function banded_sparse(N::Int, bandwidth::Int)

    est_nnz = N * bandwidth
    rows = Vector{Int}(undef, 0); sizehint!(rows, est_nnz)
    cols = Vector{Int}(undef, 0); sizehint!(cols, est_nnz)
    vals = Vector{Float64}(undef, 0); sizehint!(vals, est_nnz)
    for i in 1:N
        for j in 0:div(bandwidth,2)
            col = i + j
            if col <= N
                push!(rows, i)
                push!(cols, col) 
                push!(vals, rand())
            end
        end
    end
    H_upper = sparse(rows, cols, vals, N, N)
    return H_upper + H_upper'
end


function threaded_fill!(X::Matrix{Float64}, value::Float64)
    @threads :static for j in 1:size(X, 2)
        @simd for i in 1:size(X,1)
            @inbounds X[i,j] = value
        end
    end
end

function threaded_fill!(X::Matrix{Float64})
    @threads :static for j in 1:size(X, 2)
        @simd for i in 1:size(X,1)
            @inbounds X[i,j] = rand()
        end
    end
end

"""
    transposed_spmm!(Y, H, X)
    Computes Y = H * X using the Transposed Batch strategy.
    Y, X are (Batch X M) , H is CSC of Transpose.
"""
function transposed_spmm!(Y::Matrix{Float64}, H::SparseMatrixCSC{Float64,Int}, X::Matrix{Float64})

    local_width = size(X, 1)

    @threads :static for r in 1:H.n
        for i in H.colptr[r]:H.colptr[r+1]-1
            c = H.rowval[i]
            v = H.nzval[i]
            @simd for k in 1:local_width
                @inbounds Y[k,r] += v * X[k,c]
            end
        end
    end
end



#--- 1. FORCE THREAD PINNING ---
println("--- THREAD PINNING SETUP ---")
pinthreads(:cores)
threadinfo(;slurm=true)
println("-----------------------------")

# --- 1. Configuration (EPYC scale) ---
# WE set N = M for a square dense matrix
const N = 1_000_000
const M = 512
const BANDWIDTH = 15 # density of the sparse matrix, ~15 nonzeros per row
const GB_PER_MATRIX = (N * M * 8) / (1024^3) # size of one dense matrix in GB


println(" --- SETUP PHASE - Dense Matrix ---")

println("Matrix: $N x $M")
println("Size: $(round(GB_PER_MATRIX, digits=2)) GB per dense matrix")


X_phys = Matrix{Float64}(undef, M, N)
Y_phys = Matrix{Float64}(undef, M, N)



threaded_fill!(X_phys) #1 argument -> fill with random values
threaded_fill!(Y_phys, 0.0) #2 arguments -> fill with specified value

GC.gc()
println("Dense matrices initialized.")
# --- 2. Configuration - Sparse Matrix ---
println("\n --- SETUP PHASE - Sparse Matrix ---")
H_logical = banded_sparse(N, BANDWIDTH)
H_logical = H_logical + H_logical' # make it symmetric

# Transpose H for the transposed SpMM
H_transposed = sparse(transpose(H_logical)) 

GC.gc()


println(" --- BENCHMARK PHASE --- ")

println("\n Warmup Run: ")
transposed_spmm!(Y_phys, H_transposed, X_phys)
GC.gc()
println("\nWarmup completed.")

t_start = time()
const ITER = 5
for _ in 1:ITER
    transposed_spmm!(Y_phys, H_transposed, X_phys)
end 
t_end = time()

# --- 5. METRICS ---
avg_time = (t_end - t_start) / ITER
nnz_H = nnz(H_transposed)

total_flops = 2 * nnz_H * M
gflops = (total_flops / avg_time) / 1e9

bytes_moved = (nnz_H * M * 8.0) + (2.0 * N * M * 8.0) # Approximate bytes moved
eff_bw = (bytes_moved / avg_time) / 1e9 # in GB


println("\n--- RESULTS ---")
println("Avg Time:   $(round(avg_time, digits=4)) s")
println("Throughput: $(round(gflops, digits=2)) GFLOPS")
println("Eff. BW:    $(round(eff_bw, digits=2)) GB/s (Approx)")

if gflops > 100
    println("\n🚀 SUCCESS: Cache Trashing Fixed.")
else
    println("\n⚠️ PROBLEM.")
end
