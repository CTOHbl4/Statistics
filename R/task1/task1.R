# Стан Илья 520, 2 вариант.
# Разделитель - запятая, статистику об отсутствующих не заполнять.

library(methods)
library(utils)

# рабочая директория.
setwd("/Users/ilia.stan/Programming/R/task1")

setOldClass("data.frame")
# Создаём класс.
setClass("cars.analyzer", contains=c("data.frame"))

# функция для записи простого агрегата для 1 задания
print.aggregate <- function(object, fmt) {
  if (is.data.frame(object) & length(colnames(object)) == 2 & all(colnames(object) == c("Group.1", "x"))) {
    
    for (idx in 1:length(object$Group.1)) {
      if (idx < length(object$Group.1)) { # sep=","
        cat(sprintf(paste(fmt, ",", sep=""), object$Group.1[idx], object$x[idx]), append=TRUE)
      } else {
        cat(sprintf(fmt, object$Group.1[idx], object$x[idx]), append=TRUE)
      }
    }
    cat("\n", append=TRUE);
  } else {
    print("Wrong function usage.")
  }
}

# формат вывода из задания: Sedan =XXXXX,XX KWatt
hp.to.Kw <- function(x) {
  return(sub("\\.", ",", sprintf("%.2f", x*0.7457)))
}

Kw.to.int <- function(x) {
  splitted <- strsplit(x, ",")[[1]]
  return(as.numeric(splitted[1]) + as.numeric(splitted[2])/100)
}

# первое задание.
setMethod("print", signature("cars.analyzer"),
  function(x){
    Makers <- unique(c(x$Make))
    
    # Для каждого производителя выводим:
    for (maker in Makers) {
      sbst <- subset.data.frame(x, (Make == maker));
      cat(maker, "[", sbst$Origin[1], "]", sep="", append=TRUE);
      cat("\n", append=TRUE);
      
      agg1 <- aggregate(sbst$Type, list(sbst$Type), length);
      print.aggregate(agg1, "%s=%s");
      
      agg2 <- aggregate(sbst$Horsepower, list(sbst$Type), min);
      agg2$x <- hp.to.Kw(agg2$x)
      print.aggregate(agg2, "%s =%s KWatt")
    }
  }
)

setGeneric("write.csv")

setMethod("write.csv",
  definition=function(...) {
    args <- list(...)
    if (!is(args$x, "cars.analyzer")) {
      # поскольку переписали write.csv => используем другую функцию, чтобы не было рекурсии.
      write.table(..., sep=",")
    } else {
      # создали временный файл с наполнением из задания.
      outfile <- tempfile()
      sink(outfile)
      print(args$x)
      sink()
  
      new.dt <- data.frame(row.names = unique(c(dt$Type)))
      
      # считали все строки.
      lines <- readLines(outfile)
      unlink(outfile)
      for (idx in seq(1, length(lines), 3)) {
        origin <- substr(lines[idx], regexpr("\\[", lines[idx])[1] + 1, nchar(lines[idx])-1)
        # добавление новых регионов
        if (!(origin %in% colnames(new.dt))) {
          new.dt[origin] <- rep(NA, length(new.dt$row.names))
        }
        # парсим строку в список
        splitted <- strsplit(lines[idx+2], " KWatt,")[[1]]
        last <- splitted[[length(splitted)]]
        splitted[length(splitted)] <- substr(last, 1, nchar(last) - 6)
        
        # обновление минимумов для выбранного региона
        for (element in splitted) {
          spl.elem <- strsplit(element, " =")[[1]]
          decoded.Kw <- Kw.to.int(spl.elem[2])
          if (is.na(new.dt[spl.elem[1], origin]) | new.dt[spl.elem[1], origin] > decoded.Kw) {
            new.dt[spl.elem[1], origin] <- decoded.Kw
          }
        }
      }
      # печать на экран по умолчанию
      if (length(args$file) == 0) {
        args$file <- ""
      }
      write.table(
        apply(new.dt, c(1, 2), function(x){if (!is.na(x)){return(paste(x, "KWatt"))}else{return(x)}}),
        sep=",",
        file=args$file,
        row.names=TRUE,
        quote=FALSE,
      )
    }
  }
)

# проверки
dt <- new("cars.analyzer", read.table("CARS.csv", header=TRUE, sep=","))

print(dt)

write.csv(x=dt, file="myout.csv")

a <- read.csv("myout.csv")

write.csv(a)
