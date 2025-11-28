using LinearAlgebra, SparseArrays, LoopVectorization, Base.Threads, Printf, Random, ThreadPinning

# --- CONFIG ---
const N = 1_000_000
const M = 128           # The "Problematic" Batch Size
const TEST_BW = 50      # Physics-like Bandwidth

println("--- DEBUG RUN (N=$N, M=$M) ---")
pinthreads(:cores)

# --- SETUP ---
function generate_banded_H(n, bw)
    est_nnz = n * bw
    rows = Vector{Int}(undef, 0); sizehint!(rows, est_nnz)
    cols = Vector{Int}(undef, 0); sizehint!(cols, est_nnz)
    vals = Vector{Float64}(undef, 0); sizehint!(vals, est_nnz)
    half_bw = div(bw, 2)
    for r in 1:n
        for d in 0:half_bw
            c = r + d
            if c <= n
                push!(rows, r); push!(cols, c); push!(vals, rand())
            end
        end
    end
    H = sparse(rows, cols, vals, n, n)
    return H + H'
end

H_store = sparse(transpose(generate_banded_H(N, TEST_BW)))
X_phys = rand(M, N)
Y_phys = zeros(M, N)
GC.gc()

println("Setup Complete. Starting Diagnostics...")

# --- TEST 1: SERIAL KERNEL (Single Core Baseline) ---
# If this is fast, the problem is Threading/MPI.
# If this is slow, the problem is the Kernel/Vectorization.
println("\n[1/3] TEST 1: Single Thread Performance...")

function serial_kernel!(Y, H, X)
    local_width = size(X, 1)
    # Standard serial loop
    for r in 1:H.n
        for i in H.colptr[r]:(H.colptr[r+1]-1)
            c = H.rowval[i]
            v = H.nzval[i]
            @turbo for k in 1:local_width
                Y[k, r] += v * X[k, c]
            end
        end
    end
end

fill!(Y_phys, 0.0)
serial_kernel!(Y_phys, H_store, X_phys) # Warmup
t_serial = @elapsed serial_kernel!(Y_phys, H_store, X_phys)
flops_serial = (2.0 * nnz(H_store) * M) / t_serial / 1e9
println("   Time: $(round(t_serial, digits=4)) s | GFLOPS: $(round(flops_serial, digits=2))")


# --- TEST 2: THREADED + NO TURBO (Isolate Vectorizer) ---
# We use @simd instead of @turbo. If this is fast, @turbo is breaking.
println("\n[2/3] TEST 2: Threaded @simd (No Turbo)...")

function simd_kernel!(Y, H, X)
    local_width = size(X, 1)
    @threads :static for r in 1:H.n
        for i in H.colptr[r]:(H.colptr[r+1]-1)
            c = H.rowval[i]
            v = H.nzval[i]
            @simd for k in 1:local_width
                @inbounds Y[k, r] += v * X[k, c]
            end
        end
    end
end

fill!(Y_phys, 0.0)
simd_kernel!(Y_phys, H_store, X_phys) # Warmup
t_simd = @elapsed simd_kernel!(Y_phys, H_store, X_phys)
flops_simd = (2.0 * nnz(H_store) * M) / t_simd / 1e9
println("   Time: $(round(t_simd, digits=4)) s | GFLOPS: $(round(flops_simd, digits=2))")


# --- TEST 3: THREADED + TURBO (The Problem?) ---
# We retry the original code to confirm the bug.
println("\n[3/3] TEST 3: Threaded @turbo (Original)...")

function turbo_kernel!(Y, H, X)
    local_width = size(X, 1)
    @threads :static for r in 1:H.n
        for i in H.colptr[r]:(H.colptr[r+1]-1)
            c = H.rowval[i]
            v = H.nzval[i]
            @turbo for k in 1:local_width
                Y[k, r] += v * X[k, c]
            end
        end
    end
end

fill!(Y_phys, 0.0)
turbo_kernel!(Y_phys, H_store, X_phys) # Warmup
t_turbo = @elapsed turbo_kernel!(Y_phys, H_store, X_phys)
flops_turbo = (2.0 * nnz(H_store) * M) / t_turbo / 1e9
println("   Time: $(round(t_turbo, digits=4)) s | GFLOPS: $(round(flops_turbo, digits=2))")