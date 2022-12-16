#------------------------------------------------------------------------------------------------------------#
# Code zu 'Lineare Regression als Verfahren der erklärbaren KI anhand eines Praxisbeispiels'                 #
# Author: Michael Lebacher                                                                                   #
# Kontakt: michael.lebacher@gmx.de
#------------------------------------------------------------------------------------------------------------#

# Seed
set.seed(42)

# Import von Packeten---- 
library(ranger)
library(rpart)
library(e1071)
library(iml)
library(coefplot)

# Definition von Funktionen ----
RSQUARE = function(y_actual,y_predict){
  cor(y_actual,y_predict)^2
}

# Pfad----
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 

# Einlesen der Daten----
data = read.csv('bike_data.csv')


# Feature Kodierung ----
data$season=factor(data$season)
data$mnth = factor(data$mnth)
data$hr = factor(data$hr)
data$weekday = factor(data$weekday)
data$holiday=factor(data$holiday)
data$workingday=factor(data$workingday)
data$weathersit=factor(data$weathersit)


# Split der Daten in Training und Validierung ----
split = sample(c(rep(0, 0.7 * nrow(data)), rep(1, 0.3 * nrow(data))))

# Trainingsdatensatz
train = data[split==0,]
# Validierungsdatensatz
val = data[split==1,]

# Schätzung des linearen Modells ----

model_linreg = lm(cnt~season
                     +mnth
                     +hr
                     +holiday
                     +weekday
                     +weathersit
                     +temp
                     +atemp
                     +hum
                     +windspeed
                    ,data=train)

pred_linreg = predict(model_linreg,val)

summary(model_linreg)

mae_lin_reg = mean(abs(pred_linreg-val$cnt)) 
mae_lin_reg_train = mean(abs(predict(model_linreg,train)-train$cnt)) 
r2_lin_reg = RSQUARE(val$cnt,pred_linreg)


coefplot(model_linreg)


mod = Predictor$new(model_linreg, data = data)

eff_wsp = FeatureEffect$new(mod, feature = "windspeed")
eff_hum = FeatureEffect$new(mod, feature = "hum")
eff_atemp = FeatureEffect$new(mod, feature = "atemp")
eff_temp = FeatureEffect$new(mod, feature = "temp")

eff_seasons <- FeatureEffect$new(mod, feature = "season")

plot(eff_wsp)
plot(eff_hum)
plot(eff_atemp)
plot(eff_temp)
plot(eff_seasons)

# Plot der Feature Attribution
c1=c(27.89, 10.28,-25.99,365.2,8.95,0,177.84*0.9,92.56*0.82,-114.37*0.37,-35.8*0)
c2=c(27.89, 10.28,-11.58,-16.14,6.39,-56.30467,177.84*0.58,92.56*0.55,-114.37*0.88,-35.8*0.36)

plot(c1,ylab=c('Feature Attribution'),xlab='Feature',pch=2,cex=1,ylim=c(min(min(c1,c2)),max(max(c1,c2))),xaxt='n')
points(c2,pch=3,cex=1)
abline(h=0,lty=2)
axis(1, at = 1:10, labels = c('constant', 'season', 'month', 'hr', 'weekday','weathersit' ,'temp','atemp', 'hum','windspeed'))

for (i in 1:length(c1)){
  col ='red'
  if (c1[i]>c2[i]){
    col='green'
  }
  segments(x0=i,y0=c1[i],x1=i,y1=c2[i],col=col,lwd=1)
}
legend('topright',legend=c('index=3520','index=3296'), pch=c(2,3),bty='n')

# Schätzungs eines Regressionbaums ----

model_reg_tree = rpart(cnt~season
                       +mnth
                       +hr
                       +holiday
                       +weekday
                       +weathersit
                       +temp
                       +atemp
                       +hum
                       +windspeed
                       ,data=train)

pred_reg_tree = predict(model_reg_tree,val)

mae_reg_tree = mean(abs(pred_reg_tree-val$cnt)) 

# Schätzung eines Random Forest ----

model_rand_forest = ranger(cnt~season
                               +mnth
                               +hr
                               +holiday
                               +weekday
                               +weathersit
                               +temp
                               +atemp
                               +hum
                               +windspeed
                                ,data=train,seed=42)

pred_rand_forest = predict(model_rand_forest,val)$predictions

mae_rand_forest = mean(abs(pred_rand_forest-val$cnt)) 


# Regression mit SVM ----

model_svm = svm(cnt~season
                    +mnth
                    +hr
                    +holiday
                    +weekday
                    +weathersit
                    +temp
                    +atemp
                    +hum
                    +windspeed
                     ,data=train)

pred_svm = predict(model_svm,val)


mae_svm = mean(abs(pred_svm-val$cnt)) 

# Schätzung mit Mittelwerten der Monate ('Regel-basiertes Modell') ----

model_rule = lm(cnt~mnth
                   ,data=train)


pred_rule = predict(model_rule, val)
mae_rule = mean(abs(pred_rule-val$cnt)) 


