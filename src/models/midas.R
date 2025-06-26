library(midasr)
library(tidyverse)
library(readr)
library(zoo)
library(lubridate)
library(tidyr)
library(yaml)

# —— Config & Load data —— 
config         <- yaml::read_yaml("src/config/cfg.yml")
data_path      <- config$paths$data_processed_long
monthly_vars   <- config$features$monthly_vars
quarterly_vars <- config$features$quarterly_vars
lags           <- config$model$midas$lags
output_file    <- config$paths$outputs$midas_preds

# 1) Load & prep data
df <- read_csv(data_path, show_col_types = FALSE)

monthly <- df %>%
  filter(Variable %in% monthly_vars) %>%
  mutate(Timestamp = floor_date(Timestamp, "month")) %>%
  group_by(Variable) %>%
  complete(Timestamp = seq(min(Timestamp), max(Timestamp), by = "month")) %>%
  fill(Value) %>%
  ungroup() %>%
  pivot_wider(names_from = Variable, values_from = Value) %>%
  arrange(Timestamp)

quarterly <- df %>%
  filter(Variable %in% quarterly_vars) %>%
  mutate(Timestamp = as.Date(as.yearqtr(Timestamp))) %>%
  distinct(Timestamp, .keep_all = TRUE) %>%
  arrange(Timestamp)

# 2) Build ts objects
cpi_ts <- ts(
  monthly$CPIAUCSL,
  start     = c(year(min(monthly$Timestamp)), month(min(monthly$Timestamp))),
  frequency = 12
)
pce_ts <- ts(
  monthly$PCEPI,
  start     = c(year(min(monthly$Timestamp)), month(min(monthly$Timestamp))),
  frequency = 12
)
unr_ts <- ts(
  monthly$UNRATE,
  start     = c(year(min(monthly$Timestamp)), month(min(monthly$Timestamp))),
  frequency = 12
)

ind_ts <- ts(
  quarterly$Value,
  start     = c(year(min(quarterly$Timestamp)), quarter(min(quarterly$Timestamp))),
  frequency = 4
)

# 3) Train/test split (80/20) on quarterly series
n       <- length(ind_ts)
n_train <- floor(config$data$train_ratio * n)
y_train <- window(ind_ts, end   = time(ind_ts)[n_train])
y_test  <- window(ind_ts, start = time(ind_ts)[n_train + 1])
n_test  <- length(y_test)

# 4) Fit your MIDAS model
model <- midas_r(
  y_train ~
    mlsd(cpi_ts, lags, y_train) +
    mlsd(pce_ts, lags, y_train) +
    mlsd(unr_ts, lags, y_train),
  data  = list(y_train = y_train, cpi_ts = cpi_ts, pce_ts = pce_ts, unr_ts = unr_ts),
  start = NULL
)
print(summary(model))

# 5) Extract coefficients
coefs <- coef(model)

# 6) Manual forecast loop
preds <- numeric(n_test)
for(i in seq_len(n_test)) {
  q_time    <- time(y_test)[i]
  q_year    <- floor(q_time)
  q_num     <- cycle(y_test)[i]
  month_end <- q_num * 3
  me_date   <- as.Date(sprintf("%d-%02d-01", q_year, month_end))
  
  xreg <- numeric(length(coefs))
  xreg[1] <- 1
  idx     <- 2
  for(series in list(cpi_ts, pce_ts, unr_ts)) {
    for(j in lags) {
      dt  <- me_date %m-% months(j)
      val <- window(series,
                    start = c(year(dt), month(dt)),
                    end   = c(year(dt), month(dt)))[1]
      xreg[idx] <- val
      idx       <- idx + 1
    }
  }
  preds[i] <- sum(coefs * xreg)
}

# 7) Gather results
results <- tibble(
  date      = as.Date(paste0(floor(time(y_test)), "-", 
                             format(3 * cycle(y_test), width = 2, flag = "0"), "-01")),
  target    = as.numeric(y_test),
  predicted = preds
)
print(results)

# 8) Export
write_csv(results, output_file)

# 9) Plot Out‐of‐Sample Target vs Predicted
library(ggplot2)
ggplot(results, aes(x = date)) +
  geom_line(aes(y = target,    color = "Target"),    size = 1) +
  geom_line(aes(y = predicted, color = "Predicted"), size = 1, linetype = "dashed") +
  scale_color_manual(name = "", values = c("Target" = "blue", "Predicted" = "red")) +
  labs(
    title = "Out‐of‐Sample: Target vs Predicted",
    x     = "Quarter",
    y     = config$features$target
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    axis.text.x     = element_text(angle = 45, hjust = 1)
  )
