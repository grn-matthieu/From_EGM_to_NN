using ThesisProject
using CUDA, BenchmarkTools, StableRNGs

cfg_path = "config/smoke_cfg_stoch.yaml"  # adapte ce chemin

rng = StableRNG(1234)

function clean_gpu()
    GC.gc()
    CUDA.reclaim()        # libère la mémoire GPU inutilisée
    CUDA.synchronize()    # s'assure que tout est terminé
end

println("=== CPU run ===")
clean_gpu()
cpu_time = @belapsed begin
    solve($cfg_path; rng = $rng, opts = (use_cuda = false,))
end
println("CPU runtime: $(round(cpu_time, digits=3)) s")


println("\n=== GPU run ===")
clean_gpu()

gpu_time = @belapsed begin
    solve($cfg_path; rng = $rng, opts = (use_cuda = true,))
end
println("GPU runtime: $(round(gpu_time, digits=3)) s")

speedup = cpu_time / gpu_time
println("\nSpeedup (CPU/GPU): ×$(round(speedup, digits=2))")
