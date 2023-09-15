#Note: The submission for this script is for the results of boostedpred1.csv on
#the competition results. That was the submission that was selected for in the 
#competition, but another model was autoselected by kaggle, but those results 
#were not reproducible, so boostedtree1.csv was our choice for competition submission. 


library(tidyverse)
library(tidymodels)
library(stringr)
library(yardstick)
library(xgboost)

train <- read.csv("train2.csv")
test <- read.csv("test2.csv")

colsToRemove <- c("id","activity_year", "legal_entity_identifier_lei",
                  "total_points_and_fees", "introductory_rate_period",
                  "multifamily_affordable_units", "ethnicity_of_co_applicant_or_co_borrower_4",
                  "ethnicity_of_co_applicant_or_co_borrower_5", "race_of_co_applicant_or_co_borrower_4",
                  "race_of_co_applicant_or_co_borrower_5")

# remove unnecessary cols
train <- train %>% 
  dplyr::select(-colsToRemove)

# make the output a factor
train$action_taken <- as.factor(train$action_taken) 

# group numeric cols together in the beginning (income, loan_amount, property_value,
# combined_loan_to_value_ratio)
train <- train %>%
  relocate(income, .before = loan_type) %>%
  relocate(loan_amount, .before = loan_type) %>%
  relocate(property_value, .before = loan_type) %>%
  relocate(combined_loan_to_value_ratio, .before = loan_type)

# TESTING DATA 
# Remove unnecessary cols
test <- test %>% 
  dplyr::select(-colsToRemove)

# Group numeric cols together in the beginning (income, loan_amount, property_value,
# combined_loan_to_value_ratio)
test <- test %>% 
  relocate(income, .before = loan_type) %>% 
  relocate(loan_amount, .before = loan_type) %>% 
  relocate(property_value, .before = loan_type) %>% 
  relocate(combined_loan_to_value_ratio, .before = loan_type)

# Downsampling Randomly from the Training Set
set.seed(10)
data_split <- initial_split(train, prop = .1, strata = action_taken)

train_sample <- training(data_split)

# Creating Model
boost_tree_model <- boost_tree(learn_rate = 0.0009) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Creating recipe
rec <- recipe(action_taken ~ ., data = train_sample)

rec <- rec %>% 
  # filtering NA's from predictors with few NA's
  step_filter(!is.na(loan_term)) %>%
  # mutating in prep for dummy encoding
  step_mutate_at(starts_with("ethnicity_of"), fn = function(x) if_else(is.na(x), 4, x)) %>%
  step_mutate_at(starts_with("race_of"), fn = function(x) if_else(is.na(x), 7, x)) %>%
  step_mutate_at(starts_with("automated_underwriting_system"), fn = function(x) if_else(is.na(x), 6, x)) %>%
  # change numeric cols into double for imputation
  step_mutate_at(loan_amount, income, combined_loan_to_value_ratio, property_value,
                 fn = function(x) as.double(x)) %>%
  # imputation of numeric variables, rolling works better because linear reg cannot
  # work with NA's, and KNN takes forever
  step_impute_roll(income, combined_loan_to_value_ratio, property_value, window = 13) %>%
  # fix state NA's
  step_mutate_at(state, fn = function(x) if_else(is.na(x), "NA", x)) %>%
  # fix age_of_applicant_62 NA's
  step_mutate_at(age_of_applicant_62, fn = function(x) if_else(is.na(x), "NA", x)) %>%
  # fix age_of_co_applicant_62 NA's
  step_mutate_at(age_of_co_applicant_62, fn = function(x) if_else(is.na(x), "NA", x)) %>%
  # filter out numeric variables that still have NA's
  step_filter(!is.na(income) & !is.na(combined_loan_to_value_ratio) & !is.na(property_value)) %>%
  # since income can be negative, it cannot log transform correctly,
  # so must shift by minimum of income so lowest would be value of 1
  step_mutate_at(income, fn = function(x) x <- abs(min(x)) + x + 1) %>%
  # transformation of numeric variables
  step_log(loan_amount, income, combined_loan_to_value_ratio, property_value) %>% 
  # change all nominal variables into characters
  step_mutate_at(all_predictors(), -loan_amount, -income, -combined_loan_to_value_ratio,
                 -property_value, fn = function(x) as.factor(x)) %>% 
  # normalize all numeric predictors
  step_normalize(all_numeric_predictors()) %>%
  # remove zero variance variables
  step_zv(all_predictors()) %>%
  # remove near zero variance variables (for LDA to sorta work better)
  step_nzv(all_predictors()) %>%
  # dummy encoding of nominal variables
  step_dummy(all_predictors(), -loan_amount, -income, -combined_loan_to_value_ratio, -property_value)

# Boosted Tree -- Optimally tuned at: learn_rate = 0.0010)
set.seed(10)

# Creating workflow
wf_bt <- workflow() %>%
  add_model(boost_tree_model) %>%
  add_recipe(rec)

# Fitting model to full training data
final_model_fit <- wf_bt %>% fit(data = train)

# Make predictions on test data
preds <- final_model_fit %>% predict(new_data = test)

# Reading in test data again to reinclude original ID's
test <- read.csv("test2.csv")

# Prediction Results
test_preds <- test %>%
  dplyr::select(id) %>%
  bind_cols(preds) %>%
  rename(action_taken = .pred_class)

# CSV of results
write_csv(test_preds, "boostedpred1.csv")



