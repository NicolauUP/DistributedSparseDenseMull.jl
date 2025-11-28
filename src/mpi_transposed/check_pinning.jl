using Base.Threads
using InteractiveUtils

function get_core_id()
    # Call Linux sched_getcpu
    ccall(:sched_getcpu, Int32, ())
end

println("--- THREAD PINNING CHECK ---")
println("Threads: $(nthreads())")

# We create an array to store where each thread is running
core_map = zeros(Int, nthreads())

@threads for i in 1:nthreads()
    core_map[i] = get_core_id()
    # Busy wait to ensure they don't migrate while others start
    sleep(0.01) 
end

println("Thread -> Core Mapping:")
# Print nicely
sorted_cores = sort(unique(core_map))
println("Unique Cores used: $(length(sorted_cores))")
println("Min Core ID: $(minimum(sorted_cores))")
println("Max Core ID: $(maximum(sorted_cores))")

if length(sorted_cores) < nthreads()
    println("\n❌ DISASTER: Some threads are sharing the same core!")
    println("   (You have $(nthreads()) threads but only using $(length(sorted_cores)) cores)")
else
    println("\n✅ SUCCESS: Every thread has a unique core.")
end

# Check distribution (Are we using both sockets?)
# On EPYC 7742, Cores 0-63 are Socket 0, 64-127 are Socket 1
socket0 = count(c -> c < 64, core_map)
socket1 = count(c -> c >= 64, core_map)

println("\nSocket Balance:")
println("   Socket 0: $socket0 threads")
println("   Socket 1: $socket1 threads")

if socket0 > 0 && socket1 > 0
    println("✅ GOOD: Spanning both sockets.")
else
    println("❌ BAD: Stuck on one socket (Half Bandwidth).")
end