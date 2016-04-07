library(tm) ## texmining
library(lda) ## the actual LDA model
library(LDAvis) # visualization library for LDA

library(parallel) # multi-core paralellization

library(data.table) # fread
library(Rmpfr) # harmonic mean maximization
library(ggplot2) # pretty plotting lib
library(reshape2) # reformatting lib for ggplot2

library(tsne) # low dimensional embedding
library(caret) # ml model wrapper lib, but only used for data transformation here

library(rbokeh) # pretty (interactive) plotting

load("ldaModels_NIPS2015_k2to25_alpha001to12.rda")
load("vocab.rda")
load("termTable.rda")
load("documents.rda")


bindToEnv <- function(bindTargetEnv=parent.frame(), objNames, doNotRebind=c()) {
  # Bind the values into environment
  # and switch any functions to this environment!
  for(var in objNames) {
    val <- get(var, envir=parent.frame())
    if(is.function(val) && (!(var %in% doNotRebind))) {
      # replace function's lexical environment with our target (DANGEROUS)
      environment(val) <- bindTargetEnv
    }
    # assign object to target environment, only after any possible alteration
    assign(var, val, envir=bindTargetEnv)
  }
}

startCluster <- function(cores=detectCores()) {
  cluster <- makeCluster(cores)
  return(cluster)
}

shutDownCluster <- function(cluster) {
  if(!is.null(cluster)) {
    stopCluster(cluster)
    cluster <- c()
  }
}



papers = fread("~/Datasets/kaggle/NIPS2015_papers/Papers.csv")
docs = papers$Abstract

D <- length(documents)  # number of documents
W <- length(vocab)  # number of terms in the vocab
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document
N <- sum(doc.length)  # total number of tokens in the data
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus

G <- 100 ## number of iterations
# alpha   ## Papers are very specific such that a low alpha is more probable.
# eta     ## The scalar value of the Dirichlet hyperparamater for topic multinomials.
k = seq(2, 25, by=1) ## The number of topics a document contains.

## We will define parameter sets for 2400 LDA models, 
## which can be trained in about 1 hour using 8 cpu threads‚
params <- expand.grid(G=G,
                      k=k,
                      alpha=seq(0.01, 12, length.out=10),
                      eta=seq(0.01, 12, length.out=10))

logLiks = lapply(models, function(L)  L$log.likelihoods[1,])
logLiks.df <- as.data.frame(logLiks)
meanLogLiks <- data.frame(Iteration=1:nrow(logLiks.df), 
                          logLikelihood=rowMeans(logLiks.df))
logLiks.df$Iteration <- rownames(logLiks.df)
colnames(logLiks.df) <- 1:ncol(logLiks.df)

molten_logLiks <- melt(logLiks.df)
colnames(molten_logLiks) <- c("Iteration", "NumberOfTopics", "logLikelihood")
molten_logLiks$Iteration <- as.numeric(molten_logLiks$Iteration)

## Plot convergence
ggplot(data=molten_logLiks) + 
  geom_line(aes(x=Iteration, y=logLikelihood, color=NumberOfTopics), alpha=0.2, show.legend = FALSE) +
  geom_line(data = meanLogLiks, aes(x=Iteration, y=logLikelihood), color="black", linetype="dashed") +
  theme_bw()


harmonicMean = function(logLikelihoods, precision=2000L) {
  llMed = median(logLikelihoods)
  as.double(llMed - log(mean(exp(-mpfr(logLikelihoods,
                                       prec = precision) + llMed))))
}

paramMatrix <- function(k, params, models) {
  topic.params <- params[which(params$k==k),]
  indexes <- as.numeric(rownames(topic.params))
  alpha <- unique(topic.params$alpha)
  eta <- unique(topic.params$eta)
  selectedModels <- models[indexes]
  logLiks = lapply(selectedModels, function(m)  m$log.likelihoods[1, ])
  harmMeanLogLiks <- sapply(logLiks, function(h) harmonicMean(h))
  m <- matrix(harmMeanLogLiks, nrow=length(alpha), ncol=length(eta))
  rownames(m) <- alpha
  colnames(m) <- eta
  return(m)
}

## contour matrices for each k over alpha and eta hyperparameters
contourMatrices <- lapply(unique(params$k), paramMatrix, params, models)
## vector of argmax(likelihood) indices over all alphas of each contour matrix
opt_alphas <- sapply(contourMatrices, function(m) which(rowSums(m)==max(rowSums(m))))
## vector of argmax(likelihood) indices over all etas of each contour matrix
opt_etas   <- sapply(contourMatrices, function(m) which(colSums(m)==max(colSums(m))))
logLiks    <- sapply(contourMatrices, function(m) max(apply(m, 1, max)))
## Combine everything in a data frame including the optimal parameters over all LDAs with 
## a fixed k and number of iterations G
opt_params <- data.frame(k=unique(params$k), 
                         alpha=names(opt_alphas), 
                         eta=names(opt_etas),
                         logLikelihood=logLiks)
opt_params

## Plot harmonic mean log likelihood over all k
ggplot(data=opt_params, aes(x=k, y=logLikelihood)) +
  geom_line() +
  xlab("k") +
  ylab("Harmonic Mean log Likelihood") +
  theme_bw()

## Estimate optimal parameter set
params <- as.data.frame(apply(params, 2, as.character))

i <- which(opt_params$logLikelihood==max(opt_params$logLikelihood))
opt_k <- as.character(opt_params[i, ]$k)
opt_alpha <- as.character(opt_params[i, ]$alpha)
opt_eta <- as.character(opt_params[i, ]$eta)
opt_model_index <- as.numeric(rownames(subset(params, (k==opt_k & 
                                                       alpha==opt_alpha & 
                                                       eta==opt_eta))))
opt_k <- as.numeric(opt_k)
opt_alpha <- as.numeric(opt_alpha)
opt_eta <- as.numeric(opt_eta)
model <- models[[opt_model_index]]

# alphaCurve <- function(k, eta, params) {
#   opt_k <- as.character(opt_params[k, ]$k)
#   opt_eta <- as.character(opt_params[k, ]$eta)
#   k <- as.character(k)
#   eta <- as.character(eta)
#   
#   alpha_model_indexes <- as.numeric(rownames(subset(params, (k==opt_k & eta==opt_eta))))
#   alpha_model_logLiks <- lapply(models[alpha_model_indexes], function(m) m$log.likelihoods[1, ])
#   alpha_model_harmMeanlogLiks <- sapply(alpha_model_logLiks, function(h) harmonicMean(h))
#   return(alpha_model_harmMeanlogLiks)
# }
# 
# alphaCurve(2, opt_eta, params)

m <- melt(contourMatrices[[i]])
colnames(m) <- c("alpha", "eta", "HarmonicMeanlogLikelihood" )

## Plot alpha, eta contour
ggplot(data=m, aes(x=alpha, y=eta, z=HarmonicMeanlogLikelihood)) +
  geom_vline(xintercept = opt_alpha, colour="red", linetype = "longdash", alpha=0.5) +
  geom_hline(yintercept = opt_eta, colour="red", linetype = "longdash", alpha=0.5) +
  stat_contour(bins=20, aes(colour = ..level..)) +
  theme_bw()

N <- 5 
top.words <- top.topic.words(model$topics, 5, by.score=TRUE)
top.words.df <- as.data.frame(top.words)
colnames(top.words.df) <- 1:opt_k

top.words.df


top.documents <- top.topic.documents(model$document_sums, 
                                     num.documents = 20, 
                                     alpha = opt_alpha)
top.documents.df <- as.data.frame(top.documents)
colnames(top.documents.df) <- 1:opt_k

top.documents.df.part <- head(top.documents.df, 10)
topic_titles <- data.frame(lapply(1:opt_k, function(k) papers[as.numeric(top.documents.df.part[ ,k]),]$Title))
colnames(topic_titles) <- 1:opt_k

topic_titles

topic.proportions <- t(model$document_sums) / colSums(model$document_sums)
topic.proportions <- topic.proportions + 0.000000001 ## Laplace smoothing

getTopic <- function(topic.vec) {
  argmax <- which(topic.vec==max(topic.vec))
  if(length(argmax)>1){
    argmax <- argmax[1]
  }
  return(argmax)
}

getTopicProportion <- function(topic.vec) {
  return(topic.vec[getTopic(topic.vec)])
}

# assign each document the topic with maximum probability
doc.topic <- unlist(apply(topic.proportions, 1, getTopic))
doc.maxTopicProportion <- unlist(apply(topic.proportions, 1, getTopicProportion))
doc.topic.words <- apply(top.words[, doc.topic], 2, paste, collapse=".")


N <- 4
tp <- topic.proportions[sample(1:dim(topic.proportions)[1], N),]
colnames(tp) <- apply(top.words, 2, paste, collapse=" ")
tp.df <- melt(cbind(data.frame(tp),
                    document=factor(1:N)),
              variable.name="topic",
              id.vars = "document")  


ggplot(data=tp.df, aes(x=topic, y=value, fill=document)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle=90, hjust=1)) +
  coord_flip() +
  facet_wrap(~ document, ncol=2) +
  theme_bw()


JSD <- function(p, q) {
  m <- 0.5 * (p + q)
  divergence <- 0.5 * (sum(p * log(p / m)) + sum(q * log(q / m)))
  return(divergence)
}

n <- dim(topic.proportions)[1]
X <- matrix(rep(0, n*n), nrow=n, ncol=n)
indexes <- t(combn(1:nrow(topic.proportions), m=2))
for (r in 1:nrow(indexes)) {
  i <- indexes[r, ][1]
  j <- indexes[r, ][2]
  p <- topic.proportions[i, ]
  q <- topic.proportions[j, ]
  X[i, j] <- JSD(p,q) 
}
X <- X+t(X)
X_dist <- sqrt(X) # compute Jensen-Shannon Distance

## Projection by multidimensional scaling
X_MDS_projected <- cmdscale(X_dist, k = 2) ## Multi dimensional scaling

worker <- function() {
  bindToEnv(objNames=c("topic.proportions", "opt_k"))
  function(perplexity) {
    error <- capture.output(tsne::tsne(topic.proportions, k=2, 
                                       initial_dims=opt_k+1, 
                                       perplexity=perplexity),
                            type="message")
    error <- unlist(strsplit(error, " "))
    error <- as.numeric(error[length(error)])
    X <- tsne::tsne(topic.proportions, k=2, 
                    initial_dims=opt_k+1, 
                    perplexity=perplexity)
    result <- list(X, error)
    return(result)
  }
}

perplexities <- seq(5, 50, by=5)
t1 <- Sys.time()
cluster <- startCluster()
X_tSNE_projections <- parLapply(cluster, perplexities, worker())
shutDownCluster(cluster)
t2 <- Sys.time()
t2 - t1

## Optimal t-SNE Projection
errors <- sapply(X_tSNE_projections, function(x) x[[2]])
j <- which(errors==min(errors))
minError <- errors[j]
X_tSNE_projected <- X_tSNE_projections[[j]][[1]]

## PCA Projection
preProc = preProcess(topic.proportions, method=c("center", "scale", "pca"))
X_PCA_projected = predict(preProc, topic.proportions)[,1:2] # PCA projection

projections <- data.frame(Topic=as.factor(doc.topic), 
                          TopWords=as.factor(doc.topic.words),
                          Proportion=doc.maxTopicProportion,
                          Title=papers$Title,
                          EventType=papers$EventType,
                          Abstract=papers$Abstract,
                          x_pca=X_PCA_projected[, 1], 
                          y_pca=X_PCA_projected[, 2],
                          x_mds=X_MDS_projected[, 1], 
                          y_mds=X_MDS_projected[, 2],
                          x_tsne=X_tSNE_projected[, 1], 
                          y_tsne=X_tSNE_projected[, 2])


tools <- c("pan", 
           "wheel_zoom", "box_zoom", 
           "box_select", "lasso_select", 
           "reset", "save")                            
## PCA Plot
pca_fig <- figure(tools=tools, title="PCA") %>%
  ly_points(x_pca, y_pca, data = projections,
            color = Topic, size = Proportion*10,
            hover = list(TopWords, Proportion, Title, 
                         EventType))
## MDS Plot
mds_fig <- figure(tools=tools, title="MDS") %>%
  ly_points(x_mds, y_mds, data = projections,
            color = Topic, size = Proportion*10,
            hover = list(TopWords, Proportion, Title, 
                         EventType))
## t-SNE Plot
tsne_fig <- figure(tools=tools, title=sprintf("t-SNE (error=%.4f)", minError)) %>%
  ly_points(x_tsne, y_tsne, data = projections,
            color = Topic, size = Proportion*10,
            hover = list(TopWords, Proportion, Title, 
                         EventType))

projList <- list(pca_fig, mds_fig, tsne_fig)
p = grid_plot(projList, ncol=2, link_data=TRUE)
p
