
# Plotting & Regression Facilities
library(MASS)
data(mammals)
attach(mammals)

species <- row.names(mammals)
x <- body ; y <- brain

plot(x,y)
# identify(x,y,species)

plot(log(x), log(y))
# identify(log(x), log(y), species)

species.lm <- lm(y~x) # Linear regression, y on x
summary(species.lm)

par(mfrow=c(2,2))
plot(x,y) ; abline(species.lm)
r <- species.lm$residuals
f <- species.lm$fitted.values
qqnorm(r) ; qqline(r)

lx <- log(x) ; ly <- log(y)
species.llm <- lm(ly~lx)
summary(species.llm)

plot(lx,ly) ; abline(species.llm)
rl <- species.llm$residuals
fl <- species.llm$fitted.values
qqnorm(rl) ; qqline(rl)

plot(f,r) ; hist(r)
plot(fl,rl) ; hist(rl)

mam.mat <- cbind(x,y,lx,ly)
cor(mam.mat)
round(cor(mam.mat),3)
par(mfrow=c(1,1))
pairs(mam.mat)

# Learning and examples credited to P.M.E.Altham
