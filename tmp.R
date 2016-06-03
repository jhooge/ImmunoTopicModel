
ks <- unique(params$k)
alpha <- unique(params$alpha)[3]
eta <- unique(params$eta)[10]
selectedModels <- lapply(ks, function(k) getModelByParams(k, alpha, eta, params, models))

term.freq <- selectedModels[[1]]$topics[1,]
d <- data.frame(word = names(term.freq),
                freq = term.freq)
d<- d[with(d, order(-freq)), ]


## Returns term frequencies sorted by decreasing frequency for each topic in model


## plots the wordcloud of the term frequencies of an lda::gibbs.model
plotWordcloud <- function(term.freq) {
  d <- data.frame(word = names(term.freq),
                  freq = term.freq)
  d<- d[with(d, order(-freq)), ]
  fig <- wordcloud(words = d$word, freq = d$freq, min.freq = 100,
                   max.words=200, random.order=FALSE, rot.per=0.35,
                   colors=brewer.pal(8, "Dark2"))
  return(fig)
}

# plot the 10 most frequent words
plotMostFrequentWords <- function(term.freq, title="") {
  df <- data.frame(word = names(term.freq),
                   freq = term.freq)
  df<- df[with(df, order(-freq)), ]
  df$word <- as.factor(df$word)
  df$word <- factor(df$word, levels=df$word[order(df$freq)], ordered=TRUE)
  
  fig <- ggplot(data=df[1:20,]) + 
          geom_bar(aes(word, freq), stat="identity", position="dodge") +
          coord_flip() + 
          labs(title = title) +
          labs(x = "", y = "Absolute Frequency") + 
          theme_bw() +
          theme(plot.title   = element_text(size=18),
                strip.text.y = element_text(size=14, angle=0),
                axis.text.x  = element_text(size=12, angle=0),
                axis.text.y  = element_text(size=12),
                axis.title.x = element_text(size=10),
                axis.title.y = element_text(size=10),          
                legend.position = "none")

  return(fig)
}

for (i in 1:length(selectedModels)) {
  for (j in 1:nrow(selectedModels[[i]]$topics)) {
    print(i,j)
    term.freq <- selectedModels[[i]]$topics[j,]
    title <- sprintf("k=%i,Topic=%i", ks[i], j)
    fig <- plotMostFrequentWords(term.freq, title)
    filename <- sprintf("figures/%i-TopicModel_Topic_%i.png", ks[i], j)
    ggsave(fig, file=filename)
  }
}