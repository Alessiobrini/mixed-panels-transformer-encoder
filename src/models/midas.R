library(midasr)
library(tidyverse)
library(readr)
library(zoo)
library(lubridate)
library(tidyr)
library(yaml)
library(glue)

# —— Config & Load data ——
config <- yaml::read_yaml("src/config/cfg.yaml")

# --- Count all variables present in the raw file, regardless of role ---
if (isTRUE(config$features$all_monthly)) {
  raw_monthly_path <- config$paths$data_raw_fred_monthly
  md_header        <- names(read_csv(raw_monthly_path, n_max = 0))
  all_md_vars      <- setdiff(md_header, "date")
  
  target_var    <- config$features$target
  monthly_vars  <- setdiff(all_md_vars, target_var)
  quarterly_vars <- target_var
  
  n_monthly   <- length(all_md_vars)
  n_quarterly <- 1
} else {
  monthly_vars   <- config$features$monthly_vars
  quarterly_vars <- config$features$quarterly_vars
  n_monthly      <- length(monthly_vars)
  n_quarterly    <- length(quarterly_vars)
}

suffix <- paste0(n_monthly, "M_", n_quarterly, "Q")
lags   <- config$model$midas$lags

# Fix filepaths from template
data_path   <- as.character(glue(config$paths$data_processed_template, suffix = suffix))
output_file <- as.character(glue(config$paths$outputs$midas_preds, suffix = suffix))

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

# Only keep available variables after pivoting
available_vars <- intersect(monthly_vars, names(monthly))

quarterly <- df %>%
  filter(Variable %in% quarterly_vars) %>%
  mutate(Timestamp = as.Date(as.yearqtr(Timestamp))) %>%
  distinct(Timestamp, .keep_all = TRUE) %>%
  arrange(Timestamp)

# 2) Build monthly ts list dynamically
monthly_ts_list <- list()
for (var in available_vars) {
  ts_obj <- ts(
    monthly[[var]],
    start = c(year(min(monthly$Timestamp)), month(min(monthly$Timestamp))),
    frequency = 12
  )
  monthly_ts_list[[var]] <- ts_obj
}

# 3) Build quarterly target ts
ind_ts <- ts(
  quarterly$Value,
  start = c(year(min(quarterly$Timestamp)), quarter(min(quarterly$Timestamp))),
  frequency = 4
)

# 4) Train/test split (80/20)
n       <- length(ind_ts)
n_train <- floor(config$data$train_ratio * n)
y_train <- window(ind_ts, end = time(ind_ts)[n_train])
y_test  <- window(ind_ts, start = time(ind_ts)[n_train + 1])
n_test  <- length(y_test)

# 4b) Optional: create AR terms for y if enabled
use_y_lags <- isTRUE(config$model$midas$use_y_lags)
ar_lags    <- config$model$midas$ar_lags
y_lagged_list <- list()

if (use_y_lags && ar_lags > 0) {
  for (i in 1:ar_lags) {
    y_lagged_list[[paste0("lag", i)]] <- stats::lag(y_train, -i)
  }
}


# 5) Fit MIDAS model using dynamic formula
terms <- list()

# Add AR terms if enabled
if (use_y_lags && ar_lags > 0) {
  terms <- c(terms, lapply(seq_len(ar_lags), function(i) as.name(paste0("lag", i))))
}

# Always add MIDAS terms
terms <- c(terms, lapply(available_vars, function(v) call("mlsd", as.name(v), lags, quote(y_train))))


formula <- as.call(c(as.name("~"), quote(y_train), Reduce(function(x, y) call("+", x, y), terms)))
formula <- eval(formula)  # Evaluate the call into an actual formula object
environment(formula) <- environment()


data_list <- c(list(y_train = y_train), y_lagged_list, monthly_ts_list)


model <- midas_r(
  formula,
  data  = data_list,
  start = NULL
)
print(summary(model))

# 6) Extract coefficients
coefs <- coef(model)

# 7) Forecast loop
preds <- numeric(n_test)
for (i in seq_len(n_test)) {
  q_time    <- time(y_test)[i]
  q_year    <- floor(q_time)
  q_num     <- cycle(y_test)[i]
  month_end <- q_num * 3
  me_date   <- as.Date(sprintf("%d-%02d-01", q_year, month_end))
  
  xreg <- numeric(length(coefs))
  xreg[1] <- 1
  idx     <- 2
  
  for (var in available_vars) {
    series <- monthly_ts_list[[var]]
    for (j in lags) {
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

# 8) Results table
results <- tibble(
  date      = as.Date(paste0(floor(time(y_test)), "-", 
                             format(3 * cycle(y_test), width = 2, flag = "0"), "-01")),
  target    = as.numeric(y_test),
  predicted = preds
)
print(results)

# 9) Export
write_csv(results, output_file)

# 10) Plot
library(ggplot2)
plot_obj <- ggplot(results, aes(x = date)) +
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

print(plot_obj)
