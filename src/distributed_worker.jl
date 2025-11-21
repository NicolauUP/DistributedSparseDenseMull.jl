using LinearAlgebra
using SparseArrays
using Distributed


const WORKER_THREADS = haskey(ENV, "JULIA_WORKER_THREADS") ? parse(Int, ENV["JULIA_WORKER_THREADS"]) : 1 
BLAS.set_num_threads(WORKER_THREADS) # Ensure single-threaded BLAS on each worker



# --- 1. Worker Storage Management --- #
const WORKER_STORAGE = Dict{Symbol, Any}()



# --- 2 . Worker Storage Operations --- #
function worker_clear_storage()
    empty!(WORKER_STORAGE)

    GC.gc()
    return nothing
end


function worker_check_storage()
    keys_list = keys(WORKER_STORAGE)
    #println("Worker storage contains $(length(keys_list)) items:")
    return nothing
end


# --- 3. Initialization functions --- #


function worker_init_H(H::SparseMatrixCSC)
    # Clear previous H if it exists
    if haskey(WORKER_STORAGE, :H)
        delete!(WORKER_STORAGE, :H)
    end

    WORKER_STORAGE[:H] = H

    GC.gc()

    #println("Initialized H matrix on worker with size: $(size(H))")
    return nothing
end


# --- 4. Slab Allocation --- #

function worker_allocate_slabs(N::Int, col_range::UnitRange{Int})

    slab_width = length(col_range)

    #Define the X matrix slab (random for testing, should be other thing)
    X_slab = rand(Float64, N, slab_width)

    #Define the Y matrix slab buffer for the result
    Y_slab = zeros(Float64, N, slab_width) 



    # Save in the local storage
    WORKER_STORAGE[:X] = X_slab
    WORKER_STORAGE[:Y] = Y_slab
    WORKER_STORAGE[:col_range] = col_range

    # Report memory usage for debugging
    mem_mb = (sizeof(X_slab) + sizeof(Y_slab)) / (1024^2)
    #println("Worker $(myid): Allocated slabsfor columns $(col_range), total size: $(round(mem_mb, digits=2)) MB")

    return nothing
end



function worker_matmul()

    H = WORKER_STORAGE[:H]
    X = WORKER_STORAGE[:X]
    Y = WORKER_STORAGE[:Y]

    # Perform the multiplication
    mul!(Y, H, X)

    return nothing
end

function worker_get_diagonal()
    Y_slab = WORKER_STORAGE[:Y]
    col_range = WORKER_STORAGE[:col_range]

    local_diag = Float64[]
    sizehint!(local_diag, length(col_range))

    for (local_col_idx, global_col_idx) in enumerate(col_range)
        val = Y_slab[global_col_idx, local_cold_idx]
        push!(local_diag, val)
    end

    return local_diag
end