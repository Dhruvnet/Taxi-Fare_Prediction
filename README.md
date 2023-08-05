# Taxi-Fare_Prediction
## Data Preprocessing:
- We started by loading the dataset and checking for any missing values or inconsistencies. We handled missing values by either imputing them or dropping rows with missing values.
We converted the data types of relevant columns to the appropriate data types, such as converting coordinates to float and converting categorical variables to the category data type.

## Data Visualization:
- We created various visualizations to explore the data and gain insights into the relationships between different features and the target variable (fare amount). Some of the visualizations included histograms, scatter plots, and box plots to understand the distributions and correlations between variables.
We also plotted a heatmap to visualize the correlation between features and identified any strong correlations that could affect model performance.

## Feature Engineering:
- We engineered new features from the existing ones, such as calculating the distance between pickup and dropoff points using the Haversine formula.
We extracted additional information from the datetime features like year, weekday, and hour to capture any temporal patterns in the data.

## Model Selection:
- We decided to use multiple regression-based models for taxi fare prediction since the target variable (fare amount) is continuous. We selected the following models:
    - _<b>Linear Regression:</b>_ A simple regression model that assumes a linear relationship between the features and the target.
    -  _<b>Random Forest Regressor:</b>_ A tree-based ensemble model that can capture non-linear relationships and handle feature interactions.
    -  _<b>Decision Tree Regressor:</b>_ We selected Decision Tree Regressor because it can handle both numerical and categorical features, captures non-linear relationships in the data, and is          easy to interpret.
    -  _<b>AdaBoost Regressor:</b>_ We used AdaBoost Regressor to boost the performance of weak learners (Decision Trees) and improve generalization by giving more weight to difficult-to-predict         samples.

## Data Splitting:
- We split the dataset into training and testing sets to train the models and evaluate their performance on unseen data.

## Model Training and Evaluation:
- We trained each model on the training data and evaluated its performance on the testing data using evaluation metrics such as mean squared error (MSE), mean absolute error (MAE), and R-squared (R2) score. By comparing the performance metrics, we selected the model with the lowest MSE and MAE and the highest R2 score as the best-performing model.
- Hyperparameter Tuning: For the ensemble models (Random Forest and Gradient Boosting), we performed hyperparameter tuning using techniques like GridSearchCV to find the best set of hyperparameters that optimize model performance.
- Different Models that we have used -->
  ### Linear Regression:
  - Linear regression is a simple and interpretable model that assumes a linear relationship between the input features and the target variable (fare amount). It fits a straight line to the data       by minimizing the sum of squared errors between the predicted and actual values.
     We selected linear regression as a baseline model to establish a starting point for prediction performance. It helps us understand how well more complex models perform compared to this            simple linear approach.

  ### Random Forest Regressor:
  - The Random Forest Regressor is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and handle non-linear relationships between features and the        target. Random forests are robust against overfitting and can capture complex interactions between features, making them suitable for handling diverse and high-dimensional datasets like the       taxi fare dataset.
  
  ### Decision Tree Regressor:
  - The Decision Tree Regressor is a non-linear regression algorithm that uses a tree-like structure to make predictions. It recursively splits the data into subsets based on the features,            creating a tree where each leaf node represents a prediction for the target variable (fare amount).
  - Decision trees are capable of capturing non-linear relationships in the data, making them suitable for datasets with complex interactions between features.
  - We used Decision Tree Regressor as another baseline model to compare its performance against linear regression and ensemble-based models. Decision trees can handle both numerical and              categorical features, making them versatile for this dataset.

  ### AdaBoost Regressor:
  - The AdaBoost Regressor is an ensemble learning method that combines weak learners (e.g., Decision Trees) to create a strong learner. It sequentially builds multiple weak learners, with each       learner focusing on the samples that were misclassified by the previous learners.
  - AdaBoost is particularly effective in improving model accuracy by giving more weight to difficult-to-predict samples, and it reduces overfitting.
  - We used AdaBoost Regressor as another ensemble-based model to further enhance the predictive performance of our models. By combining multiple Decision Trees with the AdaBoost algorithm, we        aimed to achieve better generalization and accuracy on the test data.
  
## Final Model Selection:
- After hyperparameter tuning, we selected the best-performing model as our final model for taxi fare prediction which was <b>Random Forest Regression</b>

## Model Deployment:
- We saved the trained model for future use, such as making predictions on new data or deploying the model in a real-world application.
