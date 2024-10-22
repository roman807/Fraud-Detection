# Fraud detection with cost-sensitive machine learning

### Summary
I am training and testing the following five models for fraud prediction on a credit card fraud data set (available on 
[Kaggle](https://www.kaggle.com/sion1116/credit-card-fraud-detection/data))
 - Logistic regression (regular)
 - Artificial Neural Network (regular)
 - Cost-sensitive Artificial Neural Network
 - Cost-classification Logistic regression 
 - Cost-classification Artificial Neural Network 
 
All models are evaluated with 5-fold cross-validation in terms of both, F1-score and cost savings

#### For a detailed description of this project, please refer to the article [here](https://towardsdatascience.com/fraud-detection-with-cost-sensitive-machine-learning-24b8760d35d9)

### Repository organization

**main.py:** Train, test and evaluate all models

**eval_results.py:** Function to evaluate results

**ANN.py:** Artificial Neural Network with custom loss function (built in Keras)

**results.ipynb:** Generate plots to visualize results

**results** folder that contains results generated by running main.py in .npy file format
