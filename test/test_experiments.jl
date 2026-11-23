using IntervalArithmetic, IntervalBoxes
using RigorousUlam2D
using CertifiedRecords
using SparseArrays
using Test

@testset "experiment stages and the drivers' records" begin
    @testset "stages" begin
        @test ulam_system("henon").name == "henon" && ulam_system("lorenz").name == "lorenz"
        @test_throws ArgumentError ulam_system("tent")
        g = grid_constants(ulam_system("henon"), 6, 77 // 256)
        @test g.k == 64 && g.h == 2 / 64 && g.area == 4
        @test g.δ >= sqrt(big(2)) * g.h && g.varρ >= big(2) * 256 / 77
        @test_throws ArgumentError grid_constants(ulam_system("henon"), 6, 1 // 3)
        @test basename(block_file("/x", "lorenz", 9; host = "ibis")) == "tree_deepen.lorenz_d9_level14_area30_xinv_t12.ibis.jls"
        # a record name is found by latest_record, and the per-configuration names are not
        dir = mktempdir()
        r = record_file(dir, "l1_mixing", RigorousUlam2D)
        p = record_file(dir, "l1_mixing.lorenz_d8_a77-256", RigorousUlam2D)
        for f in (r, p); open_record(f, "# test\n", ["x"]); end
        @test latest_record(dir, "l1_mixing") == r
        @test latest_record(dir, "l1_mixing.lorenz_d8_a77-256") == p
        # the stage is noisy_operator of the permuted block
        sys = ulam_system("lorenz"; bisect = 12)
        blk = ulam_matrices(UlamTree(sys.F, sys.domain, 2, 4; enclosure_method = zeroth_order_enclosure,
                                     area = sys.area), 4; level = 6)
        P = noisy_ulam_operator(sys, blk, 4, 3 // 8)
        perm = grid_order(sys.domain, 4)
        _, info = noisy_operator(blk.L_lo[perm, perm], blk.U[perm, perm], SquareNoise(sys.domain, 4, 0.375; slots = 1))
        x = rand(P.n)
        @test P.info.Amul(x) == info.Amul(x) && P.normP >= 1
    end

    @testset "the drivers, end to end on a small Lorenz tree" begin
        root = mktempdir(); data = joinpath(root, "data")
        proj = pkgdir(RigorousUlam2D)
        drv(name, env) = run(addenv(`$(Base.julia_cmd()) --startup-file=no --project=$proj $(joinpath(proj, "bench", name))`,
                                    env..., "DATAROOT" => root, "DATADIR" => data))
        lor = ("MAP" => "lorenz",)
        drv("tree_deepen.jl", (lor..., "D0" => "2", "D1" => "6", "STORE" => "6", "LEVEL" => "7", "BISECT" => "12"))
        # the drivers read the blocks under the default name, level 14, bisect 30, depth 12 targets
        for d in 4:6
            cp(block_file(data, "lorenz", d; level = 7, bisect = 12, d1 = 6), block_file(data, "lorenz", d))
        end
        for d in (5, 6)
            drv("stationary_density.jl", (lor..., "DEPTH" => "$d", "A" => "3//8"))
        end
        drv("l1_mixing.jl", (lor..., "DEPTH" => "5", "A" => "3//8", "M" => "40"))
        drv("l1_transfer.jl", (lor..., "DEPTH" => "5", "A" => "3//8", "FINE" => "6"))
        _, dens = read_record(latest_record(data, "stationary_density"))
        @test length(dens) == 2 && all(r -> parse(Float64, r["eps2"]) < 1e-12, dens)
        _, pw = read_record(latest_record(data, "l1_mixing.lorenz_d5_a3-8"))
        @test length(pw) == 40 && all(r -> parse(Float64, r["C_refined"]) <= parse(Float64, r["C"]), pw)
        _, tr = read_record(latest_record(data, "l1_transfer"))
        @test length(tr) == 1 && tr[1]["fine_depth"] == "6" && isfinite(parse(Float64, tr[1]["err"]))
        # a second run of a recorded configuration does nothing
        drv("l1_transfer.jl", (lor..., "DEPTH" => "5", "A" => "3//8", "FINE" => "6"))
        @test length(last(read_record(latest_record(data, "l1_transfer")))) == 1
    end
end
