
# One-way ANOVA, qqnorm plotting
library(MASS)

# Creating data set as ragged array
x <- c(36, 36, 36, 54, 54, 54, 72, 72, 72, 108, 108, 108, 144, 144, 144)
y <- c(7.62, 8.00, 7.93, 8.14, 8.15, 7.87, 7.76, 7.73, 7.74, 7.17, 7.57, 7.80, 7.46, 7.68, 7.21)

plot(x,y)

regr <- lm(y~x) ; summary(regr)

potash <- factor(x)
plot(potash, y) # boxplot

boxregr <- lm(y~potash)
anova(boxregr)
names(boxregr)
coefficients(boxregr)

qqnorm(resid(boxregr))
qqline(resid(boxregr))
plot(fitted(boxregr), resid(boxregr))
plot(boxregr, ask = T)

# Learning and examples credited to P.M.E.Altham
