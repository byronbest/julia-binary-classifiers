module ready_data

using CSV, DataFrames, Mmap, RData, HDF5

export meta, pedigree, rows, cols, mat

"ERROR: LoadError: syntax: type declarations on global variables are not yet supported"
function __init__()
    global meta, rows, cols, mat
    "Gleason look-up: celfile_name => Gleason Score"
    meta = Dict{String,Integer}() # for each col
    "source dataset parallel to cols celfile_name"
    pedigree = Array{String,1}()
    "for checking, and because there is no way to name rows/cols in Arrays"
    rows = nothing # feature rownames
    cols = Array{String,1}() # celfile_name colnames
    mat = nothing # 2D features x celfiles
    nothing
end

"Read metadata, drop missing, and rename columns consistently."
function read_meta(dir)
    global meta
    csv1 = CSV.read(joinpath(dir,"meta.csv"),missingstring="NA")[[:celfile_name,:pathgs]]
    dropmissing!(csv1)
    csv2 = CSV.read(joinpath(dir,"Mayo.csv"))[[Symbol("CEL.file"),:pathGS]]
    rename!(csv2,f=>t for (f,t)=zip(names(csv2),names(csv1)))
    for r in eachrow(vcat(csv1,csv2))
        meta[r[1]] = r[2]
    end
end

"Read exprs, dropping rows that are superfluous."
function read_expr(dir,desc_filename,ds)
    global meta, pedigree, rows, cols, mat
    """
    desc_filename must be prepared in R to be attributes of
    bigmemory.description, as a list
    """
    df = load(joinpath(dir,desc_filename))
    "df is an artifact of the R routine that converts object to array of one"
    df = df["df"]["description"]
    println("Reading ",df["filename"])
    "assume type is double/Float64"
    println("Array of ",df["type"])
    nrow = Integer(df["nrow"]);println(nrow," feature",nrow!=1 ? "s" : "")
    ncol = Integer(df["ncol"]);println(ncol," celfile",ncol!=1 ? "s" : "")
    exprs = nothing
    open(joinpath(dir,df["filename"]),"r") do io
        try
            exprs = Mmap.mmap(io,Array{Float64,2},(nrow,ncol))
        finally
            close(io)
        end
    end

    r = df["rowNames"]
    if rows==nothing
        rows = r
    else
        "check that every exprs includes the same features"
	@assert all(rows .== r) "features do not match!"
    end
    
    celfiles = df["colNames"]
    keep = [ x in keys(meta) for x in celfiles ]
    c = celfiles[keep]
    ncol = size(c,1);println(ncol," found in meta")
    for x in c push!(cols,x); push!(pedigree,ds) end
    view(exprs,:,keep)
end

"Prepare meta by dropping Gleason Scores of 7, and combine exprs that have a score."
function read_data(dir)
    global meta, mat
    "get lookup celfile_name => Gleason Score"
    read_meta(dir)
    "Gleason Score of 7 is ambivalent"
    filter!(x->x[2]!=7,meta)

    emc = read_expr(dir,"188_desc.rda","emc")
    dkfz = read_expr(dir,"197_desc.rda","dkfz")
    mskcc = read_expr(dir,"251_desc.rda","mskcc")
    mayo = read_expr(dir,"95_desc.rda","mayo")

    "copy exprs views. Might be nice to have array of views"
    mat = hcat(emc,dkfz,mskcc,mayo)
end

function write_data(dir)
    global meta, pedigree, rows, cols, mat
    open(joinpath(dir,"gs.csv"),"w") do file
        gs=[ meta[c] for c in cols ]
        df=DataFrame([ cols, gs, pedigree ], [:cel,:gs,:ds])
        CSV.write(file,df)
        close(file)
    end
    h5open(joinpath(dir,"desc.h5"),"w") do file
        @write file rows
        @write file cols
        close(file)
    end
    open(joinpath(dir,"mat.bin"),"w") do file
        write(file,mat)
        close(file)
    end
end

"Prepare data."
function run(dir="gene")
    @time read_data(dir)
    write_data(dir)
end

end
