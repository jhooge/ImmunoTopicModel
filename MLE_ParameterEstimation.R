library(Rmpfr)

load("immunoTopicModels.rda")
load("vocab.rda")
load("records.rda")
load("termTable.rda")
load("documents.rda")

params <- expand.grid(G=100,
                      k=seq(2, 20, by=1),
                      alpha=seq(0.01, 0.5, length.out=10),
                      eta=seq(0.01, 0.5, length.out=10))
harmonicMean = function(logLikelihoods, precision=2000L) {
  llMed = median(logLikelihoods)
  as.double(llMed - log(mean(exp(-mpfr(logLikelihoods,
                                       prec = precision) + llMed))))
}

argmax <- function(x) {
  which(x==max(x))
}

argmin <- function(x) {
  which(x==min(x))
}

## Compute argmax over 
mat_argmax <- function(X) {
  row_argmax <- argmax(apply(X, 1, max))
  col_argmax <- argmax(apply(X, 2, max))
  result <- c(row_argmax, col_argmax)
  return(result)
}

## Compute maximum likelihood for model likelihood matrix with k topics
mat_max <- function(X) {
  result <- X[mat_argmax(X)[1], mat_argmax(X)[2]]
  return(result)
}

## Computes the harmonic mean of log Likelihoods over all iterations
## in an LDA model for a fixed number of topics k and returns the 
## length(params[, 1]) x length(params[, 2]) matrix
## TODO: Store the log L values in a 3 dim array, 
## LMatrix <- array(0=c(length(unique(params$k)),
##                      length(unique(params$alpha)),
##                      length(unique(params$eta))))
LMatrix <- function(k, params, models) {
  topic.params <- params[which(params$k==k),]
  indexes <- as.numeric(rownames(topic.params))
  alpha <- unique(topic.params$alpha)
  eta <- unique(topic.params$eta)
  selectedModels <- models[indexes]
  logLiks = lapply(selectedModels, function(m)  m$log.likelihoods[1, ])
  harmMeanLogLiks <- sapply(logLiks, function(h) harmonicMean(h))
  m <- matrix(harmMeanLogLiks, nrow=length(alpha), ncol=length(eta))
  return(m)
}

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

getModelByParams <- function(k, alpha, eta, params, models) {
  k_ <- as.character(k)
  alpha_ <- as.character(alpha)
  eta_ <- as.character(eta)
  params$k <- as.character(params$k)
  params$alpha <- as.character(params$alpha)
  params$eta <- as.character(params$eta)

  param_subset <- subset(params, k==k_ & alpha==alpha_ & eta==eta_)
  i <- rownames(param_subset)
  return(models[[i]])
}

## Maximum Likelihood Estimation of k, alpha and eta
LMats <- lapply(unique(params$k), LMatrix, params, models)

# save(LMats, file="LMats.rda")
load("LMats.rda")

max_LMats <- sapply(LMats, mat_max)
argmax_LMats <- lapply(LMats, mat_argmax)

opt_L <- max(max_LMats)
opt_k <- unique(params$k)[argmax(max_LMats)]
opt_alpha <- unique(params$alpha)[argmax_LMats[[argmax(max_LMats)]][1]]
opt_eta <- unique(params$eta)[argmax_LMats[[argmax(max_LMats)]][2]]

sprintf("Best model with log likelihood L=%.2f is parameterized with (k=%i, alpha=%.2f, eta=%.2f)",
        opt_L, opt_k, opt_alpha, opt_eta)

## Discussion:
## MLE for parameter estimation, does not seem to be possible:
## a) Likelihood for alpha is monotonically increasing for optimal eta for all k
## b) Likelihood for optimal eta is more or less constant for all eta and k
## c) Likelihood for k is monotonically decreasing for optimal alpha and eta.
## Model selection is mainly driven by the choice of k. With this optimization
## approach k will always be small and alpha large. So the LDA will try to classify
## each document with the smallest number of clusters, while alpha is selected as large
## as possible. A large alpha, however means that the LDA tries to explain each topic with
## the maximum number of topics available. This leads to problems in t-SNE clustering

## Let's fix eta to opt_eta and keep alpha relatively small and plot their 2d embeddings
## via t-SNE projections
## We will minimize the error of the t-SNE projectons over all perplexity
## values between 5-70 and project the documents based on the minimal error.
## after 100 iterations.

alpha <- as.numeric(unique(params$alpha)[1])
eta <- opt_eta

## Plot alpha section through hypecube at k=opt_k=2 and opt_eta=1.3222. opt_alpha=12
eta_section <- data.frame(eta=params$eta, logLikelihood=LMats[[opt_k]][2, ])
ggplot(data=eta_section, aes(x=eta, y=logLikelihood, color=logLikelihood)) +
  geom_line() +
  theme_bw()

## Plot eta section through hypercube at opt_k=2 and opt_alpha=12
alpha_section <- data.frame(alpha=params$alpha, logLikelihood=LMats[[opt_k]][, 1])
ggplot(data=alpha_section, aes(x=alpha, y=logLikelihood, color=logLikelihood)) +
  geom_line() +
  theme_bw()

## Document similarity plot for small alpha and fixed eta over all k
## Dimensionality reduction techniques used: PCA, MDS, t-SNE
pcaFig <- function(k, alpha, eta, params, models) {

  ks <- as.numeric(unique(params$k))
  selectedModels <- lapply(ks, getModelByParams, alpha, eta, params, models)
  model <- selectedModels[[k-1]]

  # assign each document the topic with maximum probability
  top.words <- top.topic.words(model$topics, 5, by.score=TRUE)
  topic.proportions <- t(model$document_sums) / colSums(model$document_sums)
  topic.proportions <- topic.proportions + 0.000000001 ## Laplace smoothing
  topic.proportions[is.na(topic.proportions)] <- 1/k
  doc.topic <- unlist(apply(topic.proportions, 1, getTopic))
  doc.maxTopicProportion <- unlist(apply(topic.proportions, 1, getTopicProportion))
  doc.topic.words <- apply(top.words[, doc.topic], 2, paste, collapse=".")

  preProc = preProcess(topic.proportions, method=c("center", "scale", "pca"))
  X_PCA_projected = predict(preProc, topic.proportions)[, 1:2] # PCA projection

  projection <- data.frame(Topic=as.factor(doc.topic),
                           TopWords=as.factor(doc.topic.words),
                           Proportion=doc.maxTopicProportion,
                           Title=ArticleTitle(records),
                           Abstract=AbstractText(records),
                           x_pca=X_PCA_projected[, 1],
                           y_pca=X_PCA_projected[, 2])

  title <- sprintf("PCA k=%i, alpha=%.2f, eta=%.2f",
                   k, alpha, eta)
  fig <- figure(title=title, title_text_font_size='10pt') %>%
    ly_points(x_pca, y_pca, data = projection,
              color = Topic, size = Proportion*10,
              hover = list(TopWords, Proportion, Title))
  return(fig)
}

tsneFig <- function(k, alpha, eta, params, models) {
  ks <- as.numeric(unique(params$k))
  opt_k=k

  selectedModels <- lapply(ks, getModelByParams, alpha, eta, params, models)
  model <- selectedModels[[k-1]]

  # assign each document the topic with maximum probability
  top.words <- top.topic.words(model$topics, 5, by.score=TRUE)
  topic.proportions <- t(model$document_sums) / colSums(model$document_sums)
  topic.proportions <- topic.proportions + 0.000000001 ## Laplace smoothing
  topic.proportions[is.na(topic.proportions)] <- 1/k
  doc.topic <- unlist(apply(topic.proportions, 1, getTopic))
  doc.maxTopicProportion <- unlist(apply(topic.proportions, 1, getTopicProportion))
  doc.topic.words <- apply(top.words[, doc.topic], 2, paste, collapse=".")

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
      result <- list(X, error, perplexity)
      return(result)
    }
  }

  perplexities <- seq(5, 70, by=5)
  t1 <- Sys.time()
  cluster <- startCluster()
  X_tSNE_projections <- parLapply(cluster, perplexities, worker())
  shutDownCluster(cluster)
  t2 <- Sys.time()
  t2 - t1

  ## Optimal t-SNE Projection
  errors <- sapply(X_tSNE_projections, function(x) x[[2]])
  j <- which(errors==min(errors))
  error <- errors[j]
  perplexity <- perplexities[j]
  X_tSNE_projected <- X_tSNE_projections[[j]][[1]]

  projection <- data.frame(Topic=as.factor(doc.topic),
                            TopWords=as.factor(doc.topic.words),
                            Proportion=doc.maxTopicProportion,
                            Title=ArticleTitle(records),
                            Abstract=AbstractText(records),
                            x_tsne=X_tSNE_projected[, 1],
                            y_tsne=X_tSNE_projected[, 2])

  title <- sprintf("t-SNE (err=%.2f, perplexity=%i) k=%i, alpha=%.2f, eta=%.2f",
                   error, perplexity, k, alpha, eta)
  title <- "t-SNE: k=12"
  fig <- figure(title=title, title_text_font_size='10pt') %>%
      ly_points(x_tsne, y_tsne, data = projection,
                color = Topic, size = Proportion*10,
                hover = list(TopWords, Proportion, Title))
  return(fig)
}
 
# ks <- params$k[1:3]
# alpha <- 0.01
# eta <- 0.01
# 
# tsneFigs <- lapply(ks, tsneFig, alpha, eta, params, models)
# pcaFigs <- lapply(ks, pcaFig, alpha, eta, params, models)
# 
# JSD <- function(p, q) {
#   m <- 0.5 * (p + q)
#   divergence <- 0.5 * (sum(p * log(p / m)) + sum(q * log(q / m)))
#   return(divergence)
# }

mdsFig <- function(k, alpha, eta, params, models) {

  ks <- as.numeric(unique(params$k))
  selectedModels <- lapply(ks, getModelByParams, alpha, eta, params, models)
  model <- selectedModels[[k-1]]

  # assign each document the topic with maximum probability
  top.words <- top.topic.words(model$topics, 5, by.score=TRUE)
  topic.proportions <- t(model$document_sums) / colSums(model$document_sums)
  topic.proportions <- topic.proportions + 0.000000001 ## Laplace smoothing
  topic.proportions[is.na(topic.proportions)] <- 1/k
  doc.topic <- unlist(apply(topic.proportions, 1, getTopic))
  doc.maxTopicProportion <- unlist(apply(topic.proportions, 1, getTopicProportion))
  doc.topic.words <- apply(top.words[, doc.topic], 2, paste, collapse=".")

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

  X_MDS_projected <- cmdscale(X_dist, k = 2) ## Multi dimensional scaling

  projection <- data.frame(Topic=as.factor(doc.topic),
                           TopWords=as.factor(doc.topic.words),
                           Proportion=doc.maxTopicProportion,
                           Title=ArticleTitle(records),
                           Abstract=AbstractText(records),
                           x_mds=X_MDS_projected[, 1],
                           y_mds=X_MDS_projected[, 2])

  title <- sprintf("MDS k=%i, alpha=%.2f, eta=%.2f",
                   k, alpha, eta)
  fig <- figure(title=title, title_text_font_size='10pt') %>%
    ly_points(x_mds, y_mds, data = projection,
              color = Topic, size = Proportion*10,
              hover = list(TopWords, Proportion, Title))
  return(fig)
}


# mdsFigs <- lapply(ks, mdsFig, alpha, eta, params, models)
# 
# ## Plot 10 random documents proportion vectors
# ## for fixed k set by cehcking t-SNE, small alpha and opt eta
# k=10
# alpha=alphas[10]
# eta=etas[2]
# model <- getModelByParams(k=k, alpha=alpha, eta=eta, params, models)
# N <- 5 
# top.words <- top.topic.words(model$topics, 5, by.score=TRUE)
# top.words.df <- as.data.frame(top.words)
# colnames(top.words.df) <- 1:opt_k
# 
# topic.proportions <- t(model$document_sums) / colSums(model$document_sums)
# topic.proportions <- topic.proportions + 0.000000001 ## Laplace smoothing
# doc.topic <- unlist(apply(topic.proportions, 1, getTopic))
# doc.maxTopicProportion <- unlist(apply(topic.proportions, 1, getTopicProportion))
# doc.topic.words <- apply(top.words[, doc.topic], 2, paste, collapse=".")
# N <- 10
# tp <- topic.proportions[sample(1:dim(topic.proportions)[1], N),]
# colnames(tp) <- apply(top.words, 2, paste, collapse=" ")
# tp.df <- melt(cbind(data.frame(tp),
#                     document=factor(1:N)),
#               variable.name="topic",
#               id.vars = "document")  
# 
# ggplot(data=tp.df, aes(x=topic, y=value, fill=document)) +
#   ggtitle(sprintf("Document Topic Proportions\n(k=%i, alpha=%.2f, eta=%.2f)", 
#                   k, alpha, eta)) +
#   geom_bar(stat="identity") +
#   theme(axis.text.x = element_text(angle=90, hjust=1)) +
#   coord_flip() +
#   facet_wrap(~ document, ncol=2) +
#   theme_bw()

# save(pcaFigs, file="mdsFigs.rda")
# save(mdsFigs, file="mdsFigs.rda")
# save(tsneFigs, file="mdsFigs.rda")
