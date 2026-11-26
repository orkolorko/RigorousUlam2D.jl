@testset "deposit_plan and write_manifest" begin
    root = mktempdir()
    data, images = mkpath(joinpath(root, "data")), mkpath(joinpath(root, "images"))
    put(dir, name, s) = (write(joinpath(dir, name), s); joinpath(dir, name))
    put(data, "d.lorenz.jls", "density bytes")
    put(data, "d.lorenz.h5", "hdf5 bytes")
    put(data, "blk_d8.jls", "coarse block")
    put(data, "blk_d12.jls", "fine block, too big"^100)
    put(data, "env.Manifest.toml", "manifest")
    put(data, "stationary_density.c0.h.tsv",
        "# bench/stationary_density.jl on h\n# RigorousUlam2D c0, julia 1.13.4, 1 threads\n" *
        "map\tdepth\ta\tblockfile\tdensity_file\n" *
        "lorenz\t12\t77//256\tblk_d12.jls\td.lorenz.jls\n" *
        "lorenz\t8\t77//256\tblk_d8.jls\tother.jls\n")
    put(data, "powers.lorenz.tsv",
        "# bench/l1_mixing.jl on h\n# map = lorenz, depth = 8, blockfile = blk_d8.jls, u = stationary_density.c0.h.tsv, η1 = 1\n" *
        "j\tC\n1\t0.5\n")
    put(data, "d9.lorenz.jls", "coarse density")
    put(data, "powers2.lorenz.tsv",
        "# bench/l1_mixing.jl on h\n# map = lorenz, depth = 9, blockfile = blk_d8.jls, u = d9.lorenz.jls\nj\tC\n1\t0.5\n")
    put(data, "l1_transfer.c0.h.tsv",
        "# bench/l1_transfer.jl on h\nmap\tSigma\tenvironment\nlorenz\t4\tenv.Manifest.toml\n")
    put(images, "figures.table_bounds.c1.h.tex", "table")
    put(images, "figures.density_lorenz_a77-256_d12.c1.h.pdf", "pdf")
    put(images, "figures.density_lorenz_a77-256_d12.c0.h.pdf", "older commit")
    sha(s) = bytes2hex(RigorousUlam2D.SHA.sha256(s))
    src = put(images, "figures.sources.c1.h.txt",
        "== l1_transfer.c0.h.tsv\n# bench/l1_transfer.jl on h\n" *
        "lorenz a = 77//256: density d.lorenz.jls, powers powers.lorenz.tsv\n" *
        "    sha256 $(sha("density bytes"))  d.lorenz.jls\n" *
        "    sha256 $(sha("hdf5 bytes"))  d.lorenz.h5\n" *
        "lorenz a = 51//256: density d.lorenz.jls, powers powers2.lorenz.tsv\n")

    p = deposit_plan(src; data, images)
    @test (p.commit, p.host) == ("c1", "h")
    @test p.rows[1].hdf5 == "d.lorenz.h5" && length(p.rows) == 2
    got = Dict(basename(f.path) => f.directory for f in p.files)
    @test got == Dict("d.lorenz.jls" => "densities", "d.lorenz.h5" => "densities",
                      "l1_transfer.c0.h.tsv" => "records", "powers.lorenz.tsv" => "records",
                      "powers2.lorenz.tsv" => "records", "d9.lorenz.jls" => "densities",
                      "stationary_density.c0.h.tsv" => "records", "env.Manifest.toml" => "environments",
                      "figures.sources.c1.h.txt" => "figures", "figures.table_bounds.c1.h.tex" => "figures",
                      "figures.density_lorenz_a77-256_d12.c1.h.pdf" => "figures")
    @test sort([basename(e.path) for e in p.excluded]) == ["blk_d12.jls", "blk_d8.jls"]
    @test all(e -> e.directory == "matrices" && startswith(e.reason, "not deposited"), p.excluded)

    pc = deposit_plan(src; data, images, blocks = :coarse)
    @test [basename(f.path) for f in pc.files if f.directory == "matrices"] == ["blk_d8.jls"]
    @test [(basename(e.path), e.reason) for e in pc.excluded] == [("blk_d12.jls", "not deposited (blocks = :coarse)")]
    pa = deposit_plan(src; data, images, blocks = :all, maxsize = 1000)
    @test [basename(f.path) for f in pa.files if f.directory == "matrices"] == ["blk_d8.jls"]
    @test [basename(e.path) for e in pa.excluded] == ["blk_d12.jls"]
    @test startswith(only(pa.excluded).reason, "1900 bytes, above the limit")

    h = write_manifest(joinpath(root, "MANIFEST.sha256"), p.files)
    lines = readlines(joinpath(root, "MANIFEST.sha256"))
    @test length(lines) == length(p.files) && issorted(last.(split.(lines, "  ")))
    @test h["densities/d.lorenz.jls"] == sha("density bytes")
    @test "$(sha("hdf5 bytes"))  densities/d.lorenz.h5" in lines

    put(data, "d.lorenz.h5", "changed")
    @test_throws ErrorException deposit_plan(src; data, images)
    put(data, "d.lorenz.h5", "hdf5 bytes")
    rm(joinpath(data, "env.Manifest.toml"))
    @test_throws ErrorException deposit_plan(src; data, images)
    @test_throws ArgumentError deposit_plan(joinpath(images, "figures.table_bounds.c1.h.tex"); data, images)
    @test_throws ArgumentError deposit_plan(src; data, images, blocks = :some)
end
