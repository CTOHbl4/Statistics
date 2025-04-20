# Стан Илья 520, вариант 2

library(nortest)
library(sasLM)
library(emmeans)
library(ggplot2)

alpha <- 0.01

setwd("/Users/ilia.stan/Programming/R/task2")
getwd()

df <- new("data.frame", read.csv("CARS.csv", header=TRUE, sep=","))
df <- na.omit(df) # всего 2 строки с пустыми значениями
attach(df)

#_________________________________________________________________________________________________
# №1
# Type -> MPG_highway

# Логично предположить, что зависит.
# Видно, что Hybrid и Sedan имеют разные расходы => зависит.
boxplot(MPG_Highway~Type)

# Оценим влияние, как неравенство 0 коэффициентов регрессии (по категориям)
summary(lm(MPG_Highway~Type))

# Видно, что все p-value малы и меньше заданного alpha.
# Следовательно, гипотеза равенства нулю коэффициентов отвергается.
# Следовательно, влияние установлено.


#_________________________________________________________________________________________________
# №2
# Преобразовать категориальные переменные для различимости категорий.

# Отметим, что различимость будет проверяться MPG_Highway~Type
# (Если, например, использовать скорость, а не расход, то результат будет другим).

# Из boxplot графика ниже можно предположить, что категории, кроме Hybrid,
# слабо различимы, если оценивать различимость как различимость средних в категориях.
# Что логично, так как Hybrid расходует топливо не так, как остальные.

# Также отмечу (надеюсь, что запомнил верно): чем выше MPG, тем меньше расход.
boxplot(MPG_Highway~Type)

# Однако требуется оценить различимости попарно, чтобы делать выводы.
Hybrid <- subset(df$MPG_Highway, (Type == "Hybrid"))
Sedan <- subset(df$MPG_Highway, (Type == "Sedan"))
Sports <- subset(df$MPG_Highway, (Type == "Sports"))
SUV <- subset(df$MPG_Highway, (Type == "SUV"))
Truck <- subset(df$MPG_Highway, (Type == "Truck"))
Wagon <- subset(df$MPG_Highway, (Type == "Wagon"))

# План:
# Проверить нормальность и одинаковость дисперсий (Т тест).
# Узнать, остался ли кто-то с симметричным распределением (Вилкоксон).
# Использовать различные критерии с уровнем alpha.


ad.test(Hybrid) # Допустим, что распределено нормально, всего 3 машины
ad.test(Sedan) # Распределено не нормально (возможно влияние "выбросов" на boxplot).
hist(Sedan, plot=TRUE) # распределение симметричное. Возможно, не хватило наблюдений. Wilcoxon test.

ad.test(Sports) # Распределено нормально
ad.test(SUV) # Распределено нормально
hist(SUV, plot=TRUE) # допустим, что нормально

ad.test(Truck) # Распределено не нормально
hist(Truck, plot=TRUE) # Распределение мультимодально. Возможно, также не хватило наблюдений.
# Отметим, что из-за мультимодальности выборки Truck, мы не можем применять к нему стандартные подходы.
# Поэтому, не будем его ни с кем объединять.
ad.test(Wagon) # Распределено нормально


# Узнаем, у кого равны дисперсии (среди нормально распределённых)
types <- list(Hybrid, Sports, SUV, Wagon)
type.names <- list("Hybrid", "Sports", "SUV", "Wagon")
df.p.values.bartlett <- data.frame(matrix(nrow=length(type.names), ncol=length(type.names)), row.names = type.names)
colnames(df.p.values.bartlett) <- type.names
for (i in 1:3)
{
  for (j in (i+1):4)
  {
    print(type.names[[i]])
    print(type.names[[j]])
    print(bartlett.test(list(types[[i]], types[[j]])))
    df.p.values.bartlett[type.names[[i]], type.names[[j]]] <-
      bartlett.test(list(types[[i]], types[[j]]))$p.value
  }
}
print(df.p.values.bartlett)
print(df.p.values.bartlett > alpha)
# Узнали, среди нормально распределённых дисперсия Sports немного отличается от Hybrid.

# Проверим неразличимость по группам:
# Группа с нормальным распр. и одинак. дисп.: (Sports, SUV, Wagon)
# Группа с не норм. распр.: (Sedan).
# Из интереса: (Truck, SUV)
# Hybrid можно ни с кем не сравнивать, так как вывод очевиден (среднее в 2 раза больше).

Norm.same.var.df <- subset(df, (Type=="Sports") | (Type=="SUV") | (Type=="Wagon"))
PDIFF(Norm.same.var.df$MPG_Highway~Norm.same.var.df$Type, Data=Norm.same.var.df, adj="tukey", conf.level=1-alpha, PLOT=TRUE)
# Видно по диффограмме и t тесту, что Wagon и Sports - неразличимы со значимостью alpha.

# Осталось сравнить Sedan с остальными.
# Sedan
Sedan.Sports <- subset(df, (Type=="Sports") | (Type=="Sedan"))
wilcox.test(MPG_Highway~Type, data=Sedan.Sports, correct=TRUE)
# различимы
Sedan.SUV <- subset(df, (Type=="SUV") | (Type=="Sedan"))
wilcox.test(MPG_Highway~Type, data=Sedan.SUV, correct=TRUE)
# различимы
Sedan.Wagon <- subset(df, (Type=="Wagon") | (Type=="Sedan"))
wilcox.test(MPG_Highway~Type, data=Sedan.Wagon, correct=TRUE)
# неразличимы

# Последнее: Truck, SUV
SUV.Truck <- subset(df, (Type=="SUV") | (Type=="Truck"))
wilcox.test(SUV.Truck$MPG_Highway~SUV.Truck$Type, data=SUV.Truck, correct=TRUE)
# Видим, что тест говорит, что вообще не различимы.
# Посмотрим на гистограммы:
hist(SUV, plot=TRUE)
hist(Truck, plot=TRUE)
# Видно, что распределения сильно различаются. Возможно, среднее Truck где-то на уровне 15.
# Так что оставим Truck отдельно.

# Итого, неразличимы следующие типы: (Wagon, Sedan, Sports). Остальные различимы.
# Объединим через изменение значений предиктора Type в df:

sample.different <- df
sample.different$Type[
  (sample.different$Type=="Sedan")|(sample.different$Type=="Wagon")|(sample.different$Type=="Sports")
  ] <- "Sedan.Wagon.Sports"

boxplot(sample.different$MPG_Highway~sample.different$Type)
# Возможно, SUV и Truck и можно объединять, но здесь оставим отдельно.

#_________________________________________________________________________________________________
# №3
# реализовать свою диффограмму.
# слой со шкалами от target + слой с диагональю + слой с решёткой по средним по категориям +
# + по слою на каждую категорию
my.diffogram <- function(target, categoric, df, alpha) {
  categories <- unique(categoric)
  categories.means <- c()
  n <- length(categories)
  
  
  for (i in 1:n)
  {
    categories.means <- append(categories.means, mean(subset(target, (categoric == categories[i]))))
  }
  ord <- sort(categories.means, decreasing = TRUE, index.return=TRUE)
  categories.means <- ord$x
  ord <- ord$ix
  categories <- categories[order(ord)]
  # сортировка гарантирует, что разность средних больше/равна 0.

  group_count <- 1
  points.df <- new("data.frame")
  # ищем концы отрезков и записываем в data.frame
  for (i in 1:(n-1))
  {
    c1 <- subset(target, (categoric == categories[i]))
    for (j in (i+1):n)
    {
      c2 <- subset(target, (categoric == categories[j]))
      
      pair.dist <- t.test(c1, c2, conf.level=1-alpha)

      # Ищем координаты концов каждого отрезка.
      center.x <- categories.means[j]
      center.y <- categories.means[i]
      
      diag_inter.x <- (center.x + center.y)/2
            
      ref.left <- pair.dist$conf.int[2]
      ref.right <- pair.dist$conf.int[1]
      real.left.x <- diag_inter.x - ref.left/2
      real.right.x <- diag_inter.x - ref.right/2
      real.left.y <- -real.left.x + center.x + center.y
      real.right.y <- -real.right.x + center.x + center.y
      type <- "different"
      if (real.right.y < real.right.x)
      {
        type <- "same"
      }
      points.df <- rbind(points.df, c(real.left.x, real.left.y, type, group_count))
      points.df <- rbind(points.df, c(real.right.x, real.right.y, type, group_count))
      group_count <- group_count + 1
    }
  }
  colnames(points.df) <- c("x", "y", "type", "group")
  points.df$x <- as.numeric(points.df$x)
  points.df$y <- as.numeric(points.df$y)
  
  a <- ggplot(points.df, aes(x, y, group = group)) + 
    geom_line(aes(color=type, linetype=type)) + geom_abline(linetype="dashed", color="grey") +
    xlim(min(min(points.df$x), min(points.df$y))-1, max(max(points.df$x), max(points.df$y))+1) + 
    ylim(min(min(points.df$x), min(points.df$y))-1, max(max(points.df$x), max(points.df$y))+1) +
    scale_color_manual(values=c('blue','red')) +
    geom_vline(xintercept = categories.means, linetype="dashed", color="grey") +
    geom_hline(yintercept = categories.means, linetype="dashed", color="grey") +
    theme(panel.background = element_blank(), panel.grid.major = element_blank())
  for (i in 1:n)
  {
    a <- a + annotate("text", x=categories.means[i], y=min(min(points.df$x), min(points.df$y)), label=categories[i])
    a <- a + annotate("text", x=max(max(points.df$x), max(points.df$y)), y=categories.means[i], label=categories[i])
  }
  a
}

my.diffogram(Norm.same.var.df$MPG_Highway, Norm.same.var.df$Type, Norm.same.var.df, alpha)


#_________________________________________________________________________________________________
# №4
# Сравнить модели без объединений неразличимых.
lm.type <- lm(MPG_Highway~Type, data=df, )
summary(lm.type)
# R^2 ~ 0.47
lm.type.origin <- lm(MPG_Highway~Type+Origin, data=df)
summary(lm.type.origin)
# R^2 ~ 0.5
# стало немного лучше. Модель улучшается.

#_________________________________________________________________________________________________
# №5
# Добавить эффект Origin*Type
lm.5 <- lm(MPG_Highway~Type+Origin+Origin*Type, data=df)
summary(lm.5)
# Выходит, что взаимодействие не даёт эффекта по сравнению с моделью без него.
# p-value значения подтверждают это (гипотеза равенства нулю коэффициентов не отвергается).

# Итоговая модель - модель с Type и Origin
summary(lm.type.origin)

my.diffogram(MPG_Highway, Type, df, alpha)
# поскольку размер выборки с Hybrid мал (3) => доверительный интервал большой.

my.diffogram(MPG_Highway, Origin, df, alpha)

#_________________________________________________________________________________________________
# №6
# Ранее было показано, что Truck значительно отличается от Sedan.
# Возможно (но вряд ли) привязка к Origin что-то изменит.
aov_model <- aov(MPG_Highway~Origin, subset(df, ((Origin!="USA")&Type=="Sedan")|(Origin=="USA"&Type=="Truck")))
plot(emmeans(aov_model, ~Origin))
# Из графика видно, что Европа и Азия сильно отличаются от Америки

# В лоб
n6 <- subset(df, ((Origin!="USA")&Type=="Sedan")|(Origin=="USA"&Type=="Truck"))
n6$Origin[n6$Origin != "USA"] <- "Other"

# Неразличимость по t тесту.
hist(subset(n6$MPG_Highway, (n6$Origin=="Other")))
hist(subset(n6$MPG_Highway, (n6$Origin=="USA")))
# Видно, что привязка к Америке не улучшила ситуацию с Truck.
# Воспользуемся тестом Вилкоксона, так как ни одна из выборок не распределена нормально.
wilcox.test(n6$MPG_Highway~n6$Origin, n6)
# Видно, что тест показывает различимость, как и ожидалось.

#_________________________________________________________________________________________________
# №7
# Модель из предыдущего номера:
model6 <- lm(n6$MPG_Highway~n6$Origin, n6)
# Рассмотрим остатки:
resid <- residuals(model6)
ad.test(resid)
# Делаем вывод, что остатки распределены не нормально.

hist(resid)
# Но, судя по гистограмме, всё не так плохо, очертания нормальности с центром в 0 имеются.
qqnorm(resid, frame=FALSE)
qqline(resid, col = "red", lwd = 2)
# Наверное, можно сказать, что это похоже на нормальное распределение.
# Итого, с натяжкой ограничение выполняется.

# Дисперсии:
bartlett.test(list(subset(n6$MPG_Highway, n6$Origin=="Other"), subset(n6$MPG_Highway, n6$Origin=="USA")))
# Тест утверждает, что гипотеза равенства дисперсий принимается.

# Вывод: в целом требования выполняются.


#_________________________________________________________________________________________________
# №8
# Тест Крускала-Уоллиса
kruskal.test(n6$MPG_Highway~n6$Origin, n6)
# Результат примерно тот же, как и в тесте Вилкоксона

#_________________________________________________________________________________________________
# №9
# txt -> pdf
# Только anova с пометкой p-value (внутри txt, отдельно).
sink("anova.txt")

print("#1")
summary(lm(MPG_Highway~Type))

print("#2")
ad.test(Hybrid)
ad.test(Sedan)
ad.test(Sports)
ad.test(SUV)
ad.test(Truck)
ad.test(Wagon)

types <- list(Hybrid, Sports, SUV, Wagon)
type.names <- list("Hybrid", "Sports", "SUV", "Wagon")
for (i in 1:4)
{
  for (j in (i+1):4)
  {
    print(type.names[[i]])
    print(type.names[[j]])
    print(bartlett.test(list(types[[i]], types[[j]])))
  }
}


PDIFF(MPG_Highway~Norm.same.var.df$Type, Data=Norm.same.var.df, adj="tukey", conf.level=1-alpha, PLOT=FALSE)

wilcox.test(MPG_Highway~Type, data=Sedan.Sports, correct=TRUE)
wilcox.test(MPG_Highway~Type, data=Sedan.SUV, correct=TRUE)
wilcox.test(MPG_Highway~Type, data=Sedan.Wagon, correct=TRUE)
wilcox.test(SUV.Truck$MPG_Highway~SUV.Truck$Type, data=SUV.Truck, correct=TRUE)

print("#4")
summary(lm.type)
summary(lm.type.origin)

print("#5")
summary(lm.5)
summary(lm.type.origin)

print("#6")
summary(aov_model)
wilcox.test(n6$MPG_Highway~n6$Origin, n6)

print("#7")
ad.test(resid)
bartlett.test(list(subset(n6$MPG_Highway, n6$Origin=="Other"), subset(n6$MPG_Highway, n6$Origin=="USA")))

print("#8")
kruskal.test(n6$MPG_Highway~n6$Origin, n6)

sink()
sink()

