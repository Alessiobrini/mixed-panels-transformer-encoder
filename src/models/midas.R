library(midasr)
library(tidyverse)
library(readr)
library(zoo)
library(lubridate)
library(tidyr)

# 1) Load & prep data
df <- read_csv("data/processed/long_format_fred.csv", show_col_types = FALSE)

monthly <- df %>%
  filter(Variable %in% c("CPIAUCSL","PCEPI","UNRATE")) %>%
  mutate(Timestamp = floor_date(Timestamp, "month")) %>%
  group_by(Variable) %>%
  complete(Timestamp = seq(min(Timestamp), max(Timestamp), by = "month")) %>%
  fill(Value) %>%
  ungroup() %>%
  pivot_wider(names_from = Variable, values_from = Value) %>%
  arrange(Timestamp)

quarterly <- df %>%
  filter(Variable == "INDPRO") %>%
  mutate(Timestamp = as.Date(as.yearqtr(Timestamp))) %>%
  distinct(Timestamp, .keep_all = TRUE) %>%
  arrange(Timestamp)

# 2) Build ts objects
cpi_ts  <- ts(monthly$CPIAUCSL,
              start     = c(year(min(monthly$Timestamp)), month(min(monthly$Timestamp))),
              frequency = 12)
pce_ts  <- ts(monthly$PCEPI,
              start     = c(year(min(monthly$Timestamp)), month(min(monthly$Timestamp))),
              frequency = 12)
unr_ts  <- ts(monthly$UNRATE,
              start     = c(year(min(monthly$Timestamp)), month(min(monthly$Timestamp))),
              frequency = 12)

ind_ts  <- ts(quarterly$Value,
              start     = c(year(min(quarterly$Timestamp)), quarter(min(quarterly$Timestamp))),
              frequency = 4)

# 3) Train/test split (80/20) on quarterly series
n       <- length(ind_ts)
n_train <- floor(0.8 * n)
y_train <- window(ind_ts, end   = time(ind_ts)[n_train])
y_test  <- window(ind_ts, start = time(ind_ts)[n_train + 1])
n_test  <- length(y_test)

# 4) Fit your MIDAS model
model <- midas_r(
  y_train ~
    mlsd(cpi_ts, 0:11, y_train) +
    mlsd(pce_ts, 0:11, y_train) +
    mlsd(unr_ts, 0:11, y_train),
  data  = list(y_train = y_train, cpi_ts = cpi_ts, pce_ts = pce_ts, unr_ts = unr_ts),
  start = NULL
)
print(summary(model))

# 5) Extract coefficients
coefs <- coef(model)
# Expect length(coefs) == 1 + 3*12 = 37

# 6) Manual forecast loop
preds <- numeric(n_test)

for(i in seq_len(n_test)) {
  # which quarter in the original series this corresponds to
  q_time   <- time(y_test)[i]                # e.g. 2012.00, 2012.25, ...
  q_year   <- floor(q_time)
  q_num    <- cycle(y_test)[i]               # 1..4
  month_end <- q_num * 3                      # Q1→3 (Mar), Q2→6, Q3→9, Q4→12
  # the Date of the 1st of that month:
  me_date  <- as.Date(sprintf("%d-%02d-01", q_year, month_end))
  
  # build the regressor vector: intercept + (cpi lags, pce lags, unr lags)
  xreg <- numeric(length(coefs))
  xreg[1] <- 1
  
  idx <- 2
  for(series in list(cpi_ts, pce_ts, unr_ts)) {
    for(j in 0:11) {
      dt  <- me_date %m-% months(j)
      # grab the one value at year(dt), month(dt)
      val <- window(series,
                    start = c(year(dt), month(dt)),
                    end   = c(year(dt), month(dt)))[1]
      xreg[idx] <- val
      idx <- idx + 1
    }
  }
  
  preds[i] <- sum(coefs * xreg)
}

# 7) Gather results
results <- tibble(
  date = as.Date(paste0(floor(time(y_test)), "-", 
                        format(3 * cycle(y_test), width = 2, flag = "0"), "-01")),
  target    = as.numeric(y_test),
  predicted = preds
)

print(results)

# 8) Export
write_csv(results, "outputs/midas_preds_3vars.csv")


# 7) Plot Out‐of‐Sample Target vs Predicted
library(ggplot2)

ggplot(results, aes(x = date)) +
  geom_line(aes(y = target,    color = "Target"),    size = 1) +
  geom_line(aes(y = predicted, color = "Predicted"), size = 1, linetype = "dashed") +
  scale_color_manual(name = "", values = c("Target" = "blue", "Predicted" = "red")) +
  labs(
    title = "Out‐of‐Sample: Target vs Predicted",
    x     = "Quarter",
    y     = "INDPRO"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    axis.text.x     = element_text(angle = 45, hjust = 1)
  )
