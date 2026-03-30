"Get FRED data and apply transformations as in 
McCracken, Michael W. and Ng, Serena (2015)
https://www.stlouisfed.org/research/economists/mccracken/fred-databases
"

devtools::install_github("cykbennie/fbi")
library(fbi)
setwd("G:/My Drive/Attention MIDAS/fred data")
data = fredmd(file='current_md.csv', transform = TRUE)
write.csv(data,file="transf_md.csv")
data = fredqd(file='current_qd.csv', transform = TRUE)
write.csv(data,file="transf_qd.csv")