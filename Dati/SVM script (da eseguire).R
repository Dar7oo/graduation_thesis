# Pre-processing
library(e1071)
library(dplyr)
library(caret)
library(factoextra)

dat5 <- read.csv("5year.csv")
dat5 <- dat5[-1]


# Rinomino le variabili
names(dat5) <- gsub(pattern = "Attr", replacement = "X", x = names(dat5))


# Codifico i "?" come valori mancanti
for(i in 1:65){
  dat5[dat5[i] == "?", i] <- NA
}

# Funzione per convertire i predittori in tipo numeric e la variabile di risposta
# in factor
to_factor <- function(dat){
  res <- dat %>%
    mutate_at(1:64, as.numeric) %>% 
    mutate_at(65, as.factor)
  
  return(res)
}

dat5 <- to_factor(dat5)

# Rimozione dei record con valori mancanti
dat5 <- dat5[complete.cases(dat5), ]

# Analisi correlazioni
cor(dat5[, 1:64]) > 0.8

# Nel dataset sembrano essere presenti predittori fortemente correlati tra loro.
# Dato che questi contengono informazioni ridondanti, ricorro all'analisi delle
# componenti principali per cercare di ridurre la dimensionalità dei dati,
# preservando allo stesso tempo almeno il 90% dell'informazione iniziale

# PCA
pc <- prcomp(dat5[, 1:64], center = TRUE)

#names(pc)

#head(pc$x)

#pc$loadings

fviz_eig(pc, addlabels = TRUE)
# Le prime tre componenti principali sono in grado di cogliere oltre il 99% della
# variabilità osservata quindi ovviamente manterrò solo queste.


l <- pc$sdev^2
cv <- cumsum(l)/sum(l)*100
#cv

plot(l/sum(l)*100, t = 'l', col = 2, lwd = 2,
     ylab = "Varianza [%]",
     xlab = "Componente")
grid()

k <- min( which(cv >= 99) )
#k

# Torno allo spazio di partenza (NOTA: lo spazio CENTRATO)
Phi <- pc$rotation
Z <- pc$x

Xk <- Z[ , 1:k] %*% t(Phi[ , 1:k])

# "de-centro" i dati sommando le medie di colonna alle rispettive colonne
mu <- pc$center
tildeX <- sweep(Xk , 2, mu , "+") 


dat <- data.frame(class = dat5$class,
                  tildeX)



############################# Costruzione modello ##############################
# Construisco l'oggetto train_control con le impostazioni necessarie per
# implementare Repeated Cross Validation. I parametri sono descritti nella tesi.

# Cambio i livelli delle classi da "0-1" a "No-Yes" altrimenti train() da errore
levels(dat$class) <- c("No", "Yes")

# Creo i folds su cui fare Repeated Cross Validation
set.seed(1)
folds <- createMultiFolds(dat$class, k = 10, times = 10)

train_control <- trainControl(method = "repeatedcv",
                              index = folds,
                              savePredictions = "all",
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE,
                              sampling = "smote",
                              verboseIter = TRUE)


# RBF kernel SVM
svm_RBF_fit <- train(class ~., data = dat, method = "svmRadial",
                     trControl = train_control, preProcess = c("center","scale"),
                     tuneLength = 5, metric = "ROC")


# Polynomial kernel SVM
# Per il valore di scale: https://bookdown.org/mpfoley1973/supervised-ml/support-vector-machines.html
svm_POLY_fit <- train(class ~., data = dat, method = "svmPoly",
                      trControl = train_control, preProcess = c("center", "scale"), metric = "ROC",
                      allowParallel = TRUE,
                      tuneGrid = expand.grid(degree = c(2, 3, 4),
                                             C = c(0.25, 0.50, 1, 2, 4),
                                             scale = 0.001))


# Linear kernel SVM
svm_LINEAR_fit <- train(class ~., data = dat, method = "svmLinear",
                        trControl = train_control, preProcess = c("center", "scale"),
                        tuneGrid = data.frame(C = c(0.25, 0.50, 1, 2, 4)), metric = "ROC")





################################################################################
# GRAFICI
# Polynomial kernel SVM performance
svm_POLY_res <- data.frame(C = svm_POLY_fit$results$C,
                           degree = svm_POLY_fit$results$degree,
                           ROC = svm_POLY_fit$results$ROC)

svm_POLY2_res <- svm_POLY_res[svm_POLY_res$degree == 2, ]
svm_POLY3_res <- svm_POLY_res[svm_POLY_res$degree == 3, ]
svm_POLY4_res <- svm_POLY_res[svm_POLY_res$degree == 4, ]

# RBF kernel performance
svm_RBF_res <- data.frame(C = svm_RBF_fit$results$C,
                          ROC = svm_RBF_fit$results$ROC)

# Linear kernel performance
svm_LINEAR_res <- data.frame(C = svm_LINEAR_fit$results$C,
                             ROC = svm_LINEAR_fit$results$ROC)

# AUC plot
C <- c(0.25, 0.50, 1, 2, 4)
plot(x = C, y = svm_LINEAR_res$ROC, t = "b", pch = 0, col = "red", lwd = 2,
     ylab = "AUC", ylim = c(0.55, 0.76))

lines(x = C, y = svm_POLY2_res$ROC, t = "b", pch = 0, col = "blue", lwd = 2)
lines(x = C, y = svm_POLY3_res$ROC, t = "b", pch = 0, col = "green", lwd = 2)
lines(x = C, y = svm_POLY4_res$ROC, t = "b", pch = 0, col = "purple", lwd = 2)

lines(x = C, y = svm_RBF_res$ROC, t = "b", pch = 0, col = "orange", lwd = 2)


title(main = "SVM kernel comparison - AUC", sub = "poly.scale = 0.01, RBF.sigma = 5.526677")
legend(x = "bottomleft", legend = c("Linear", "Poly d=2", "Poly d=3", "Poly d=4", "RBF"),
       col = c("red", "blue", "green", "purple", "orange"), lty = 1, lwd = 2, cex = 0.8)
grid()


################################################################################
# Curva ROC
library(pROC)

# https://stackoverflow.com/questions/37244383/r-get-auc-and-plot-multiple-roc-curves-together-at-the-same-time

# Seleziono gli indici da usare per prelevare le previsioni dei migliori settaggi di ogni kernel per quanto riguarda l'iperparametro C
idx_RBF <- which(svm_RBF_fit$pred$C == 0.25)
idx_POLY <- which(svm_POLY_fit$pred$C == 1)
idx_LINEAR <- which(svm_LINEAR_fit$pred$C == 0.25)


# Visualizzo le curve ROC dei modelli con kernel lineare, RBF e polinomiale di quarto grado
plot.roc(svm_RBF_fit$pred$obs[idx_RBF], svm_RBF_fit$pred$No[idx_RBF],
         col = "orange", axes = TRUE,
         print.thres = TRUE, print.thres.adj = c(1.025, -0.5),
         print.auc = TRUE, auc.polygon = TRUE,
         print.auc.x = 1, print.auc.y = 0.98, asp = 0.8,
         main = "Confronto funzioni kernel - ROC")


plot.roc(svm_POLY_fit$pred$obs[idx_POLY], svm_POLY_fit$pred$No[idx_POLY],
         col = "purple",
         print.thres = FALSE,
         print.auc = TRUE, 
         print.auc.x = 1, print.auc.y = 0.93,
         add = TRUE)


plot.roc(svm_LINEAR_fit$pred$obs[idx_LINEAR], svm_LINEAR_fit$pred$No[idx_LINEAR],
         col = "red",
         print.thres = FALSE,
         print.auc = TRUE, 
         print.auc.x = 1, print.auc.y = 0.88,
         add = TRUE)
grid()






################################################################################
# Prelevo gli Precision e Recall da ogni modello
#get_f1 <- function(svm_fit){
#  to_return <- matrix(data = rep(0, 100), nrow = 10)
#  
#  for(i in 1:10){
#    for(j in 1:10){
#      # Compongo la stringa per prelevare i risultati desiderati
#      str1 <- sprintf("%02d", i)
#      str2 <- sprintf("%02d", j)
#      
#      fold <- paste0("Fold", str1, ".", "Rep", str2)
#      #print(fold)
#      
#      # Seleziono i risultati
#      idx <- svm_fit$pred$Resample == fold
#      svm_fold <- svm_fit$pred[idx, ]
#      
#      # Matrice di confusione
#      #   - positive specifica la classe da considerare come positiva
#      #   - mode = "prec_recall" fa in modo che venga restituito lo score F1
#      confmat <- caret::confusionMatrix(data = svm_fold$pred,
#                                 reference = svm_fold$obs,
#                                 positive = "Yes", mode = "prec_recall")
#
#      to_return[i, j] <- confmat$byClass[7]
#    }
#  }
#  return(to_return)
#}
#
#rec_LINEAR <- get_rec(svm_LINEAR_fit)
#rec_POLY <- get_rec(svm_POLY_fit)
#rec_RBF <- get_rec(svm_RBF_fit)





# Visualizzo i risultati
#plot(x = 1:100, y = rec_LINEAR, t="l", col = "red", lwd = 2,
#     main = "Mean F-score", xlab = "index", ylab = "Recall")
#abline(h = mean(rec_LINEAR), col = "red", lwd = 2)
#
#lines(x = 1:100, y = rec_POLY, t="l", col = "purple", lwd = 2)
#abline(h = mean(rec_POLY), col = "purple", lwd = 2)
#
#lines(x = 1:100, y = rec_RBF, t="l", col = "orange", lwd = 2)
#abline(h = mean(rec_RBF), col = "orange", lwd = 2)
#
#grid()
#legend(x = "topleft", legend = c("Linear", "Poly d = 4", "RBF"),
#       col = c("red", "purple", "orange"), lwd = 2)






################################################################################
# Salvataggio grafici
# Impostazioni
ar <- sqrt(2)
h <- 110
hh <- h / 25.4
ww <- hh * ar



# PCA screeplot
pdf(file = "screeplot.pdf", height = hh, width = ww,  pointsize = 8)
fviz_eig(pc, addlabels = TRUE, main = "")
dev.off()




# Iperparametri funzioni kernel e AUC
pdf(file = "auc_performance.pdf", height = hh, width = ww,  pointsize = 8)
plot(x = C, y = svm_LINEAR_res$ROC, t = "b", pch = 0, col = "red", lwd = 2,
     ylab = "AUC", ylim = c(0.55, 0.76))

lines(x = C, y = svm_POLY2_res$ROC, t = "b", pch = 0, col = "blue", lwd = 2)
lines(x = C, y = svm_POLY3_res$ROC, t = "b", pch = 0, col = "green", lwd = 2)
lines(x = C, y = svm_POLY4_res$ROC, t = "b", pch = 0, col = "purple", lwd = 2)

lines(x = C, y = svm_RBF_res$ROC, t = "b", pch = 0, col = "orange", lwd = 2)


title(main = "Confronto funzioni kernel - AUC", sub = "poly.scale = 0.01, RBF.sigma = 5.526677")
legend(x = "bottomleft", legend = c("Linear", "Poly d=2", "Poly d=3", "Poly d=4", "RBF"),
       col = c("red", "blue", "green", "purple", "orange"), lty = 1, lwd = 2, cex = 0.8)
grid()
dev.off()




# Curve ROC funzioni kernel
pdf(file = "roc_performance.pdf", height = hh, width = ww,  pointsize = 8)
plot.roc(svm_RBF_fit$pred$obs[idx_RBF], svm_RBF_fit$pred$No[idx_RBF],
         col = "orange", axes = TRUE,
         print.thres = TRUE, print.thres.adj = c(1.025, -0.5),
         print.auc = TRUE, auc.polygon = TRUE,
         print.auc.x = 1, print.auc.y = 0.98, asp = 0.8,
         main = "Confronto funzioni kernel - ROC")


plot.roc(svm_POLY_fit$pred$obs[idx_POLY], svm_POLY_fit$pred$No[idx_POLY],
         col = "purple",
         print.thres = FALSE,
         print.auc = TRUE, 
         print.auc.x = 1, print.auc.y = 0.93,
         add = TRUE)


plot.roc(svm_LINEAR_fit$pred$obs[idx_LINEAR], svm_LINEAR_fit$pred$No[idx_LINEAR],
         col = "red",
         print.thres = FALSE,
         print.auc = TRUE, 
         print.auc.x = 1, print.auc.y = 0.88,
         add = TRUE)
grid()
dev.off()


