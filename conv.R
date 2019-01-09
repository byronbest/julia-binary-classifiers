library(bigmemory)

data <- readRDS("gene/norm_Mayo-Jenkins-MetsDisc_95.bin.desc")
df <- as.list(attributes(data))
save(df,file="gene/95_desc.rda")

data <- readRDS("gene/norm_EMC-Boormans-External_188.bin.desc")
df <- as.list(attributes(data))
save(df,file="gene/188_desc.rda")

data <- readRDS("gene/norm_DKFZ-Brase-External_197.bin.desc")
df <- as.list(attributes(data))
save(df,file="gene/197_desc.rda")

data <- readRDS("gene/norm_MSKCC-Taylor-External_251.bin.desc")
df <- as.list(attributes(data))
save(df,file="gene/251_desc.rda")
