# Databricks notebook source
# MAGIC %md
# MAGIC # modeling utilities
# MAGIC
# MAGIC **Objective: This notebook defines python functions that we use for the modeling purposes and also the ann model class is also defined in this notebook..** <br>
# MAGIC   
# MAGIC ---
# MAGIC **Process:**
# MAGIC * Import used packages
# MAGIC * Define the following functions "apply_feature_clipping", "apply_min_max_scaling", "generate_train_test_validation_data_loaders", "calculate_mae_divided_by_mean_actual_values", "train_ann_model".

# COMMAND ----------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

# COMMAND ----------

def apply_feature_clipping(feature_space : pd.DataFrame, list_of_features : list, what_percentage_to_clip : float = 0.999) -> (pd.DataFrame, dict):
  """
  This function normalizes our input feature space. In feature clipping, we drop the very extreme values through feature clipping -- cutting off the top 0.001 values-.

  Parameters
  --------
  feature_space: Pandas.DataFrame
    A panda table that contains all the feature space including training, testing, and validation sets
  list_of_features: list of str
    A list of feature or features that we want to apply the clipping to. 
  what_percentage_to_clip: float
    A float number that tells us what perentage of top values should be dropped

  Returns
  --------
  DataFrame:
    DataFrame that is feature clipped normalized.
  dict
    A dictionary containing the clipped values

  """
  # We use the following dictionary to filter the baseline model to make sure that we compare apples to apples, meaning when defining our baseline models, only consider the rows that were used for modeling and testing too.
  clipped_values_dictionary = {}

  #A very skewed dataset can really get in the way of training a DL model. So, we drop the 0.1 percentile (1/10 of 1%) of all the top past-yeras profit values. This approach is called feature clipping.
  for feature_name in list_of_features:
    print(f"we are analyzing column {feature_name}")
    print(f"We started with {len(feature_space)} rows")
    number_of_unique_values = len(feature_space[feature_name].unique())

    #get the 0.999 % percentile index and value
    nine_nity_nine_percentile_index = round(what_percentage_to_clip * number_of_unique_values)
    nine_ninty_nine_percentile_value = (sorted(feature_space[feature_name].unique())[nine_nity_nine_percentile_index:][0])
    print(f"Values bigger than {nine_ninty_nine_percentile_value} will be dropped for {feature_name} column, but this is not the value we will have in 'clipped_values_dictionary' because we will be dropping rows by other features too")

    #Now, we drop all the rows that are bigger than "nine_ninty_nine_percentile_value"
    how_may_rows_are_we_dropping = len(feature_space[feature_space[feature_name] >
                                                                  nine_ninty_nine_percentile_value])
    print(f"We will be dropping {how_may_rows_are_we_dropping} rows")
    feature_space = feature_space[feature_space[feature_name] <= nine_ninty_nine_percentile_value]
    print(f"we end up having {len(feature_space)} rows")
    print("--" * 10)

  for feature_name in list_of_features:
    features_max_value = feature_space[feature_name].max()
    print(f"The final value in 'clipped_values_dictionary' for column {feature_name} is {features_max_value}")
    clipped_values_dictionary[feature_name] = features_max_value

  return feature_space, clipped_values_dictionary

# COMMAND ----------

def apply_min_max_scaling(feature_space: pd.DataFrame, list_of_features : list, min_and_max_values_dictionary: dict) -> pd.DataFrame:
  """
  This function normalizes our input feature space. In min-max sclipping, we limit the range of our values between 0 and 1.

  Parameters
  --------
  feature_space: Pandas.DataFrame
    A panda table that contains all the feature space including training, testing, and validation sets
  list_of_features: list of str
    A list of feature or features that we want to apply the clipping to. 
  min_and_max_values_dictionary: dictionary of float value
    A dictionary containing the min and max values of features to be scaled
    

  Returns
  --------
  DataFrame:
    DataFrame that min-max scaling is applied.

  """
  for feature_name in list_of_features:
    column_max_value = min_and_max_values_dictionary[f"{feature_name}_MAX_VALUE"]
    column_min_value = min_and_max_values_dictionary[f"{feature_name}_MIN_VALUE"]
    print(f"We are working on column {feature_name}, and the max value is {column_max_value}")

    feature_space[feature_name] = (feature_space[feature_name] - column_min_value) / (column_max_value - column_min_value)
    print(f"We are done with column {feature_name}, and now the max value is {feature_space[feature_name].max()}")

  
  return feature_space

# COMMAND ----------

def generate_train_test_validation_data_loaders(input_feature_space: pd.DataFrame, 
                                                independent_columns: list,
                                                features_to_be_min_max_scaled :list,
                                                identifier_columns: list = ["MATERIAL_ID", "SALES_OFFICE_DESC", "MVP_STORE_TYPE"],
                                                dependent_column: str = "PROFIT", 
                                                train_validation_size: int = 10000, 
                                                validation_size: int = 5000, 
                                                training_batch_size: str = 4096) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, pd.DataFrame) :
  """
  This function takes in a feature space and gives us back tensor data loader that can be used to train our neural network model.

  Parameters
  --------
  input_feature_space: panda.DataFrame
    It is a panda dataframe containing all the data that we have and has both dependent and independent values inside.
  identifier_columns: list of str, default ["MATERIAL_ID", "SALES_OFFICE_DESC", "MVP_STORE_TYPE"]
    A list that contains the identfifier column meaning, this is the granulairty of our prediction meaning we want the profit prediction for this combination
  dependent_column: str, default "PROFIT"
    A string that specifies the dependent value that are trying to predict
  independent_columns: list str
    A list containing name of column or columns that are used as the predicting informations
  features_to_be_min_max_scaled: list of str
    A list that contains the name of the features that we will be scaling
  train_validation_size: int
    An integer value that tells us the total count of rows dedicated to testing and validation
  validation_size: int
    An integer value that tells us the total count of rows dedicated to validation
  training_batch_size: str, default 4096
    An integer that specifies the size of the training batches

  Returns
  ------
  three torch.utils.data.DataLoader, the order of the output is as follows: training, testing, and validation 
  validation_feature_space: pd.DatFrame
    We need this to be able to pinpoint cases where the biggest error happens.
  training_set_min_max_values_dictonary
    We use this values to bring back scaled values to their normal values
  
  """
  import warnings
  warnings.filterwarnings("ignore")

  actual_values = input_feature_space[identifier_columns + [dependent_column]]
  feature_space = input_feature_space[identifier_columns + independent_columns]

  
  (training_feature_space, testing_validation_feature_space, 
  training_actual_values, testing_validation_actual_values) = train_test_split(feature_space, actual_values, 
                                                                            test_size = train_validation_size, shuffle = True
                                                                            , random_state=1) 
  assert len(testing_validation_feature_space) == train_validation_size, "The test-validation chunk size is not what we intended to have"

  #Get the min and max value fo features we want to min-max scaled in the training set. We then use those values to scale features in test and validation set
  training_set_min_max_values_dictonary = {}
  for feature_name in features_to_be_min_max_scaled:
    feature_max_value = training_feature_space[feature_name].max()
    feature_min_value = training_feature_space[feature_name].min()
    training_set_min_max_values_dictonary[f"{feature_name}_MAX_VALUE"] = feature_max_value
    training_set_min_max_values_dictonary[f"{feature_name}_MIN_VALUE"] = feature_min_value


  (testing_feature_space, validation_feature_space, 
  testing_actual_values, validation_actual_values) = train_test_split(testing_validation_feature_space, 
                                                                    testing_validation_actual_values, test_size = validation_size, shuffle = True,
                                                                    random_state=1)
  
  print(f"The max value for feature 'LAST_YEAR_PROFIT' in table is training_feature_space {training_feature_space['LAST_YEAR_PROFIT'].max()}")
  print(f"The max value for feature 'LAST_YEAR_PROFIT' in table is testing_feature_space {testing_feature_space['LAST_YEAR_PROFIT'].max()}")
  print(f"The max value for feature 'LAST_YEAR_PROFIT' in table is validation_feature_space {validation_feature_space['LAST_YEAR_PROFIT'].max()}")

  training_feature_space_min_max_scaled = apply_min_max_scaling(training_feature_space, features_to_be_min_max_scaled, training_set_min_max_values_dictonary)
  print("Done min_max_scaling training set")
  print("**" * 60)
  testing_feature_space_min_max_scaled = apply_min_max_scaling(testing_feature_space, features_to_be_min_max_scaled, training_set_min_max_values_dictonary)
  print("Done min_max_scaling testing set")
  print("**" * 60)
  validation_feature_space_min_max_scaled = apply_min_max_scaling(validation_feature_space, features_to_be_min_max_scaled, training_set_min_max_values_dictonary)
  print("Done min_max_scaling validation set")
  print("**" * 60)

  training_feature_space_tensor = torch.tensor(training_feature_space_min_max_scaled[independent_columns].values).float()
  training_actual_values_tensor = torch.tensor(training_actual_values[dependent_column].values).float().reshape(-1, 1)

  testing_feature_space_tensor = torch.tensor(testing_feature_space_min_max_scaled[independent_columns].values).float()
  testing_actual_values_tensor = torch.tensor(testing_actual_values[dependent_column].values).float().reshape(-1, 1)

  validation_feature_space_tensor = torch.tensor(validation_feature_space_min_max_scaled[independent_columns].values).float()
  validation_actual_values_tensor = torch.tensor(validation_actual_values[dependent_column].values).float().reshape(-1, 1)


  training_tensor_dataset = TensorDataset(training_feature_space_tensor, training_actual_values_tensor)
  testing_tensor_dataset = TensorDataset(testing_feature_space_tensor, testing_actual_values_tensor)
  validation_tensor_dataset = TensorDataset(validation_feature_space_tensor, validation_actual_values_tensor)

  training_data_loader = DataLoader(training_tensor_dataset, training_batch_size, drop_last = True)
  testing_data_loader = DataLoader(testing_tensor_dataset, batch_size = len(testing_tensor_dataset.tensors[0]))
  validation_data_loader = DataLoader(validation_tensor_dataset, batch_size = len(validation_tensor_dataset.tensors[0]))

  print(f"We have {len(training_data_loader)} of training batches and the batch size is {training_data_loader.batch_size}")
  print(f"We have {len(testing_data_loader)} of testing batches and the batch size is {testing_data_loader.batch_size}")
  print(f"We have {len(validation_data_loader)} of validation batches and the batch size is {validation_data_loader.batch_size}")

  return training_data_loader, testing_data_loader, validation_data_loader, validation_feature_space, training_set_min_max_values_dictonary

# COMMAND ----------

class AnnModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers_dictionary = nn.ModuleDict()

    self.layers_dictionary["INPUT_LAYER"] = nn.Linear(3, 256)

    self.layers_dictionary["HIDDEN_LAYER_1"] = nn.Linear(256, 256)
    self.layers_dictionary["BATCH_NORMALIZER_HIDDEN_LAYER_1"] = nn.BatchNorm1d(256)

    self.layers_dictionary["HIDDEN_LAYER_2"] = nn.Linear(256, 128)
    self.layers_dictionary["BATCH_NORMALIZER_HIDDEN_LAYER_2"] = nn.BatchNorm1d(256)

    self.layers_dictionary["HIDDEN_LAYER_3"] = nn.Linear(128, 128)
    self.layers_dictionary["BATCH_NORMALIZER_HIDDEN_LAYER_3"] = nn.BatchNorm1d(128)

    self.layers_dictionary["HIDDEN_LAYER_4"] = nn.Linear(128, 128)
    self.layers_dictionary["BATCH_NORMALIZER_HIDDEN_LAYER_4"] = nn.BatchNorm1d(128)

    self.layers_dictionary["HIDDEN_LAYER_5"] = nn.Linear(128, 64)
    self.layers_dictionary["BATCH_NORMALIZER_HIDDEN_LAYER_5"] = nn.BatchNorm1d(128)

    self.layers_dictionary["HIDDEN_LAYER_6"] = nn.Linear(64, 64)
    self.layers_dictionary["BATCH_NORMALIZER_HIDDEN_LAYER_6"] = nn.BatchNorm1d(64)

    self.layers_dictionary["OUTPUT_LAYER"] = nn.Linear(64, 1)

  def forward(self, input_feature_space, batch_normalization_is_active):

    input_layer_output = F.relu(self.layers_dictionary["INPUT_LAYER"](input_feature_space))

    if batch_normalization_is_active == True:
      hidden_layer_1_input_normalized = self.layers_dictionary["BATCH_NORMALIZER_HIDDEN_LAYER_1"](input_layer_output)
      hidden_layer_1_output = F.relu(self.layers_dictionary["HIDDEN_LAYER_1"](hidden_layer_1_input_normalized))

      hidden_layer_2_input_normalized = self.layers_dictionary["BATCH_NORMALIZER_HIDDEN_LAYER_2"](hidden_layer_1_output)
      hidden_layer_2_output = F.relu(self.layers_dictionary["HIDDEN_LAYER_2"](hidden_layer_2_input_normalized))
      
      hidden_layer_3_input_normalized = self.layers_dictionary["BATCH_NORMALIZER_HIDDEN_LAYER_3"](hidden_layer_2_output)
      hidden_layer_3_output = F.relu(self.layers_dictionary["HIDDEN_LAYER_3"](hidden_layer_3_input_normalized))

      hidden_layer_4_input_normalized = self.layers_dictionary["BATCH_NORMALIZER_HIDDEN_LAYER_4"](hidden_layer_3_output)
      hidden_layer_4_output = F.relu(self.layers_dictionary["HIDDEN_LAYER_4"](hidden_layer_4_input_normalized))

      hidden_layer_5_input_normalized = self.layers_dictionary["BATCH_NORMALIZER_HIDDEN_LAYER_5"](hidden_layer_4_output)
      hidden_layer_5_output = F.relu(self.layers_dictionary["HIDDEN_LAYER_5"](hidden_layer_5_input_normalized))

      hidden_layer_6_input_normalized = self.layers_dictionary["BATCH_NORMALIZER_HIDDEN_LAYER_6"](hidden_layer_5_output)
      hidden_layer_6_output = F.relu(self.layers_dictionary["HIDDEN_LAYER_6"](hidden_layer_6_input_normalized))
    
    else:
      hidden_layer_1_output = F.relu(self.layers_dictionary["HIDDEN_LAYER_1"](input_layer_output))
      hidden_layer_2_output = F.relu(self.layers_dictionary["HIDDEN_LAYER_2"](hidden_layer_1_output))
      hidden_layer_3_output = F.relu(self.layers_dictionary["HIDDEN_LAYER_3"](hidden_layer_2_output))
      hidden_layer_4_output = F.relu(self.layers_dictionary["HIDDEN_LAYER_4"](hidden_layer_3_output))
      hidden_layer_5_output = F.relu(self.layers_dictionary["HIDDEN_LAYER_5"](hidden_layer_4_output))
      hidden_layer_6_output = F.relu(self.layers_dictionary["HIDDEN_LAYER_6"](hidden_layer_5_output))
    
    output_layer_output = self.layers_dictionary["OUTPUT_LAYER"](hidden_layer_6_output)

    return output_layer_output

# COMMAND ----------

def train_ann_model(ann_model_class: AnnModel(), 
                    number_of_epochs: int, 
                    training_data_loader: torch.utils.data.DataLoader, 
                    testing_data_loader: torch.utils.data.DataLoader, 
                    batch_normalization_is_active: bool,
                    learning_rate: float = 0.01) -> (list, AnnModel(), list, list):
  """
  This function helps us train our ANN model. The process of forward propagation, backward propagation happens inside this function, and we also record the perofrmance of the learned parameters for each epoch.

  Parameters
  --------
  ann_model_class: class
    An class of ann model containing all the information about our ann model including number of layers, number of nodes, batch normalization, drop out and so on
  number_of_epochs: int
    An integer that specifies the number of training epochs that we go through. This number in addition to the training batch size are the main items determing the time it takes to train our model. The higher the epoch number and smaller the batch size, the longer it takes to train our model
  training_data_loader: torch.utils.data.DataLoader
    A DatLoader that holds our training set
  testing_data_loader: torch.utils.data.DataLoader
    A DatLoader that holds our testing set
  batch_normalization_is_active: bool
    This parameter specifies whether to use batch normalization or not.
  learning_rate: float
    A float which specifies the learning rate of our ANN model.

  Return
  --------
  loss_list: list
    A list containing the training losses
  trained_model: torch.nn.module
    The trained model 
  list_of_training_error_value: list
    A list containing training error
  list_of_testing_error_value: list
    A list containing testing error
  """
  trained_model = ann_model_class
  #Choosing our loss and optimizer
  loss_function = nn.L1Loss()
  optimization_function = torch.optim.Adam(trained_model.parameters(), lr = learning_rate)

  loss_list = []
  list_of_training_error_value = []
  list_of_testing_error_value = []

  for epoch_number in range(number_of_epochs):
    print(f"we are in epoch {epoch_number + 1}")

    batches_loss_list = []
    batches_error_value_list = []
    #batch training starts here
    for feature_space, actual_values in training_data_loader:
      #forward propagation
      prediction = trained_model(feature_space, batch_normalization_is_active)
      loss_value = loss_function(prediction, actual_values)
      #Backward propagation
      optimization_function.zero_grad()
      loss_value.backward()
      optimization_function.step()

      batches_loss_list.append(loss_value.detach().item())
      batches_error_value_list.append(calculate_mae_divided_by_mean_actual_values(prediction, actual_values))

    #Evaluating the model by measuring test accuracy
    trained_model.eval()
    testing_feature_space, testing_actual_values = next(iter(testing_data_loader))
    with torch.no_grad():
      testing_prediction = trained_model(testing_feature_space, batch_normalization_is_active)
      testing_error = calculate_mae_divided_by_mean_actual_values(testing_prediction, testing_actual_values)
    
    #Saving the evaluation values of each epoch
    loss_list.append(np.mean(batches_loss_list))
    print(f"The training loss value for epoch number {epoch_number + 1} is {np.mean(batches_loss_list)}")
    list_of_training_error_value.append(np.mean(batches_error_value_list))
    list_of_testing_error_value.append(testing_error)

  return loss_list, trained_model, list_of_training_error_value, list_of_testing_error_value

# COMMAND ----------

def calculate_mae_divided_by_mean_actual_values(predicion_tensor: torch.tensor, actual_values_tensor: torch.tensor) -> float:
  """
  This function helps us calculate MAE(Mean Absolute Error) / Mean Actual values. It gives us a percentage that can be interpreted as percentage of error when predicting the mean case.

  Arguments
  --------
  predicion_tensor: torch.tensor
    A tensor containing all the prediction values 
  actual_values_tensor: torch.tensor
    A tensor containing all the actual values that we are trying to predict

  Returns
  --------
  A float
    A float value that can be interpreted as the error
  """
  absolute_error = torch.sum(torch.abs(predicion_tensor - actual_values_tensor))
  mean_absolute_error =  absolute_error.item() / len(actual_values_tensor)
  mean_absolute_error_divided_by_mean_actual_values = mean_absolute_error / torch.mean(actual_values_tensor).item()

  return mean_absolute_error_divided_by_mean_actual_values

# COMMAND ----------

# MAGIC %md
# MAGIC ###### =======================================================================================================
# MAGIC #### The importing process was successful. We can continue
# MAGIC ###### =======================================================================================================