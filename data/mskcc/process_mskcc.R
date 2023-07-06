# process_mskcc.R

library(limma)

# load data
load(url('https://github.com/gdancik/BC-BET/raw/main/setup/data/processed/mskcc.RData'))
load(url('https://github.com/gdancik/BC-BET/raw/main/setup/data/clinical/mskcc.RData'))

# write files
write.table(mskcc_clinical,  col.names = TRUE, row.names = FALSE,
          file = 'mskcc_clinical.csv', quote = FALSE, sep = '\t')

write.table(mskcc.expr,  col.names = TRUE, row.names = TRUE,
            file = 'mskcc_expr.csv', quote = FALSE, sep = '\t')

# find differentially expressed genes

tumor <- mskcc_clinical$tumor
design <- model.matrix(~-1+tumor)
colnames(design) <- c("normal", "tumor")

# 'lmFit' fits a linear model to each row of the expression matrix ##
fit <- lmFit(mskcc.expr, design)

# Specify the contrasts -- the names here must match column names of 
# design matrix 
contrast.matrix <- makeContrasts(tumor - normal,levels=design)

## fit model based on contrasts (e.g., tumor - normal)
fit <- contrasts.fit(fit, contrast.matrix)

# calculate moderated t-statistics by moderating standard errors
# toward the expected value, using limma trend. 
fit.de <- eBayes(fit)

# get the top differentially expressed probes, 
# sorted by p-value ('topTable' gives top 10 by default)
tt <- topTable(fit.de, sort.by = "p", p.value = 0.01, number = Inf)
tt

m1 <- match(rownames(tt), rownames(mskcc.expr))
m2 <- match(rownames(tt)[1:30], rownames(mskcc.expr))


write.table(mskcc.expr[m1,],  col.names = TRUE, row.names = TRUE,
            file = 'mskcc_expr_fdr_1.csv', quote = FALSE, sep = '\t')

write.table(mskcc.expr[m2,],  col.names = TRUE, row.names = TRUE,
            file = 'mskcc_expr_top30.csv', quote = FALSE, sep = '\t')


# look at heatmap


# extract expression values for DE probes

expr <- mskcc.expr[m2,]

# create a color range consisting of 200 values between yellow and blue
col.heat <- colorRampPalette(c("yellow", "blue"))(200)

# set colors for gender #
col <- as.integer(as.factor(tumor))
col <- c("lightblue", "darkred")[col]

# Generate the heatmap
heatmap(expr, ColSideColors = col, 
        col = col.heat, scale = "row")

