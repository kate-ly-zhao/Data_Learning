
# Pi with Monte Carlo

sim <- 10000
perimeter <- 1

# Randomly generating points
rand_pt <- function(perimeter = 1) {
  x <- runif(n = 1, min = -perimeter, max = perimeter)
  y <- runif(n = 1, min = -perimeter, max = perimeter)
  return(list(x = x, y = y, in_circle = x^2 + y^2 <= perimeter^2))
}

# Monte Carlo 
set.seed(123)
pi_df <- data.frame(x = rep(NA, sim), y = rep(NA, sim), in_circle = rep(NA, sim))
for (i in seq(sim)) {
  my_sim <- rand_pt()
  pi_df$in_circle[i] <- my_sim$in_circle
  pi_df$x[i] <- my_sim$x
  pi_df$y[i] <- my_sim$y
}

my_pi <- 4 * sum(pi_df$in_circle) / nrow(pi_df)

library(ggplot2)
ggplot(pi_df, aes(x=x, y=y, color=as.factor(in_circle))) + geom_point() + theme(legend.position = 'none')

# Credit: Alook Analytics
