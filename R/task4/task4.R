# Стан Илья 520, 2 вариант
library(nortest)
library(emmeans)
library(multcomp)
library(pls)
library(olsrr)
library(ggplot2)
library(pracma)


setClass("my.glm", slots=c("stats", "meanHP", "meanW", "meanCyl", "frequentType",
                           "predictors", "mod"))


# Отмечу, что по условию fit зависит только от train_data,
# Но тогда не ясно, как заполнять слоты объекта.
# В условии task3 в fit был параметр объект.
# Добавим и здесь.
setGeneric("fit", function(model, train_data) {
  standardGeneric("fit")
})
setGeneric("mape", function(model, data) {
  standardGeneric("mape")
})
setGeneric("prepare_data", function(model, data, verbous = TRUE) {
  standardGeneric("prepare_data")
})
setGeneric("predict", function(model, test_data, strict = TRUE, verbous = TRUE) {
  standardGeneric("predict")
})
setGeneric("plot", function(model, nice = FALSE) {
  standardGeneric("plot")
})

setMethod("prepare_data",
          "my.glm",
          
          function(model, data, verbous = TRUE) {
            # Заполнение пропусков средними значениями
            if (verbous) {
              print("Проверка заполнения пропусков")
              print(any(is.na(data)), length(data))
            }
            data$Type[(is.na(data$Type))] = model@frequentType
            data$Cylinders[(is.na(data$Cylinders))] = model@meanCyl
            data$Horsepower[(is.na(data$Horsepower))] = model@meanHP
            data$Weight[(is.na(data$Weight))] = model@meanW
            if (verbous) {
              print(any(is.na(data)), length(data))
            }
            
            # объединение Sedan и Wagon, так как неразличимы
            data$Type[(data$Type == "Sedan")] = "SW"
            data$Type[(data$Type == "Wagon")] = "SW"
            
            # Взятие функции от предикторов (после заполнения пропусков)
            data$LogHorsepower <- log(data$Horsepower)
            data$LogWeight <- log(data$Weight)
            return(data[model@predictors])
          }
)

setMethod("fit",
          "my.glm",
          
          function(model, train_data) {
            # Убираем по одному предиктору итерационно,
            # смотря на значения статистик.
            # Затем пробуем добавлять преобразования.
            
            # Итого я получил следующие предикторы:
            model@predictors <- c("LogWeight", "Type", "LogHorsepower", "Cylinders")
            
            # На обучении убираю выбросы (расчёт на то, что в обучающей выборке их немного)
            train_data <- na.omit(train_data)

            # Можно объединить следующие значения категор. переменных
            train_data$Type[(train_data$Type == "Sedan")] <- "SW"
            train_data$Type[(train_data$Type == "Wagon")] <- "SW"
            
            # Запомним параметры для заполнения пропусков
            # Можно обучать простые регрессии и классификации, но ограничимся средними.
            model@meanHP <- round(mean(train_data$Horsepower))
            model@meanW <- round(mean(train_data$Weight))
            model@frequentType <- names(sort(table(train_data$Type),decreasing=TRUE)[1])
            model@meanCyl <- round(mean(train_data$Cylinders))

            # Запомним числа для plot
            model@stats <- list("Sports"=c(), "SW"=c(), "SUV"=c(), "Truck"=c(), "Hybrid"=c())
            d <- subset(train_data, Type == "Sports")
            model@stats$Sports[1] <- min(d$Weight)
            model@stats$Sports[2] <- max(d$Weight)
            model@stats$Sports[3] <- min(d$Horsepower)
            model@stats$Sports[4] <- max(d$Horsepower)
            model@stats$Sports[5] <- round(mean(d$Cylinders))
            d <- subset(train_data, Type == "SW")
            model@stats$SW[1] <- min(d$Weight)
            model@stats$SW[2] <- max(d$Weight)
            model@stats$SW[3] <- min(d$Horsepower)
            model@stats$SW[4] <- max(d$Horsepower)
            model@stats$SW[5] <- round(mean(d$Cylinders))
            d <- subset(train_data, Type == "SUV")
            model@stats$SUV[1] <- min(d$Weight)
            model@stats$SUV[2] <- max(d$Weight)
            model@stats$SUV[3] <- min(d$Horsepower)
            model@stats$SUV[4] <- max(d$Horsepower)
            model@stats$SUV[5] <- round(mean(d$Cylinders))
            d <- subset(train_data, Type == "Truck")
            model@stats$Truck[1] <- min(d$Weight)
            model@stats$Truck[2] <- max(d$Weight)
            model@stats$Truck[3] <- min(d$Horsepower)
            model@stats$Truck[4] <- max(d$Horsepower)
            model@stats$Truck[5] <- round(mean(d$Cylinders))
            d <- subset(train_data, Type == "Hybrid")
            model@stats$Hybrid[1] <- min(d$Weight)
            model@stats$Hybrid[2] <- max(d$Weight)
            model@stats$Hybrid[3] <- min(d$Horsepower)
            model@stats$Hybrid[4] <- max(d$Horsepower)
            model@stats$Hybrid[5] <- round(mean(d$Cylinders))
            
            print("Минимальный и максимальный вес и мощность по категориям:")
            print(model@stats)
            
            train_data$LogHorsepower <- log(train_data$Horsepower)
            train_data$LogWeight <- log(train_data$Weight)

            X <- train_data[model@predictors]
            # Промежуточная модель
            mod = glm(train_data$MPG_City~., data = X, family = poisson(link = "log"))
            
            # Убрали объект (один, если обучать на всём cars.csv), дающий
            # слишком большой остаток.
            train_data <- subset(train_data, mod$residuals < 0.4)

            X <- train_data[model@predictors]
            # Итоговая модель
            model@mod = glm(train_data$MPG_City~., data = X, family = poisson(link = "log"))
            
            hist(model@mod$residuals, breaks = 10)
            
            # тест признаёт нормальность остатков (если обучать на всём cars.csv).
            print(ad.test(model@mod$residuals))
            print(summary(model@mod))
            
            
            base::plot(model@mod$linear.predictors, model@mod$residuals)
            
            print("MAPE на тренировочной:")
            print(mean(abs(round(predict.glm(model@mod, type = "response")) - train_data$MPG_City)/train_data$MPG_City))
            
            return(model)
          }
)

# отклик целочисленный, так как целевая переменная - целочисленная.
setMethod("predict",
          "my.glm",
          
          function(model, test_data, strict = TRUE, verbous = TRUE) {
            # подготовка данных + обработка пропусков
            X <- prepare_data(model, test_data, verbous)
            if (strict){
              return(round(predict.glm(model@mod, newdata = X, type = "response")))
            }
            return(predict.glm(model@mod, newdata = X, type = "response"))
          }
)

setMethod("plot",
          "my.glm",
          
          function(model, nice = FALSE) {
            # Контурный график
            # Не совсем конкретно сказано, что значит усреднение переменной.
            # Также не совсем понятно, как получится сетка 20 на 20, если одна из переменных
            # категориальная, и число категорий меньше 20.
            
            # Возьмём непрерывные предикторы и построим сетку 20 на 20 на них.
            # Построим контурный график для каждого типа кузова.
            
            for (t in c("Sports", "SW", "SUV", "Truck", "Hybrid")) {
              cont <- model@stats[[t]]
              
              W <- linspace(cont[1], cont[2], 20)
              HP <- linspace(cont[3], cont[4], 20)
              z <- c()
              
              for (i in W) {
                for (j in HP) {
                  z[length(z)+1] <- predict(model, list("Weight"=c(i),
                                                    "Type"=c(t),
                                                    "Horsepower"=c(j),
                                                    "Cylinders"=c(cont[5])), strict = !nice, verbous = FALSE)
                }
              }
              contour(W, HP, matrix(z, 20, 20), title = title(t), xlab = "Weight",
                      ylab = "Horsepower", col=10)
            }
          }
)

setMethod("mape",
          "my.glm",
          
          function(model, data) {
            return(mean(abs(predict(model, data) - data$MPG_City)/data$MPG_City))
          }
)


# _________________________________________


setwd("/Users/ilia.stan/Programming/R/task4")
getwd()

train_data <- new("data.frame", read.csv("CARS.csv", header=TRUE, sep=","))

object <- new("my.glm")

object <- fit(object, train_data)

train_data$MPG_City - predict(object, train_data)

mape(object, train_data) # добавили "проблемный" объект и два объекта с пропусками.

plot(object)


# ________________
# Пара комментариев:
# В обучении пропуски убираются, на тесте пропуски заполняются средними значениями
# в методе prepare_data.

# Если из итоговых предикторов убрать цилиндры, то ad.test = 0.166 (много).
# Однако, тогда качество становится чуть хуже. Выбор сделан в пользу качества.
# Из контурных графиков видно, что чем меньше MPG, тем меньше становится представителей.
# (График становится пологим, чем меньше MPG, тем больше HP и Weight).

# В задании сказано строить график отклика, а у меня берётся округление. Поэтому
# график немного ломаный. При необходимости увидеть гладкий график следует переключить
# флаг nice (необязательный параметр).