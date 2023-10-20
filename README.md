# MLAI_Module17 - Comparing Classifiers - Practical Application III

## Overview
In this practical application, our goal is to compare the performance of different classifiers, specifically K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. We'll use a dataset related to marketing bank products over the telephone and this repository contains the code, analysis and details related to this dataset.

The dataset we're working with comes from the UCI Machine Learning repository [here](https://archive.ics.uci.edu/ml/datasets/bank+marketing) and contains results from multiple marketing campaigns conducted by a Portuguese banking institution. To understand the dataset in more detail, you can refer to the article accompanying the dataset [here](CRISP-DM-BANK.pdf).

## Table of Contents

- [MLAI\_Module17 - Comparing Classifiers - Practical Application III](#mlai_module17---comparing-classifiers---practical-application-iii)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Understanding the Data](#understanding-the-data)
  - [Reading the Data](#reading-the-data)
  - [Understanding the Features](#understanding-the-features)
  - [Understanding the Task](#understanding-the-task)
  - [Feature Engineering](#feature-engineering)
  - [Train/Test Split](#traintest-split)
  - [Baseline Model](#baseline-model)
  - [Simple Model](#simple-model)
  - [Model Comparisons with default settings](#model-comparisons-with-default-settings)
  - [Improving the Model and the comparisons](#improving-the-model-and-the-comparisons)
  - [Recommendations](#recommendations)
  - [Data Source](#data-source)
  - [References](#references)

## Understanding the Data

The data we have comes from 17 marketing campaigns that happened between May 2008 and November 2010. These campaigns were carried out by the bank, and in total, they involved contacting 79,354 people.

From the article's "Experiments and Results" section reveals that the study involved three main phases of testing on the marketing campaign data:

**First Phase:** In the initial stage, the researchers used the Naive Bayes algorithm on the original data, which had 12 possible output values. They were able to create a basic model, but its performance was limited.

**Second Phase:** In the next stage, they refined the output to a simple classification, categorizing outcomes as either successful or unsuccessful. They tested this modified data with both Naive Bayes and Decision Tree algorithms, which improved performance, although there was still room for enhancement.

**Third Phase:** In the final phase, they conducted a thorough analysis that reduced the input features from 59 to 29. They also removed instances with missing values. This time, they tested the Naive Bayes, Decision Trees, and SVM (Support Vector Machine) algorithms. The SVM algorithm yielded the best results, achieving an AUC (Area Under the Curve) of 0.938 and an ALIFT (AUC Lift) of 0.887.

**Summary**:

  * The study comprised three distinct testing phases.
  * In the third phase, they evaluated the performance of Naive Bayes, Decision Tree, and SVM models.
  * Among these models, the SVM model stood out as the most accurate in predicting the success of marketing contacts.
  * These experiments showcase a systematic approach (CRISP-DM) to improving model performance by understanding the data, streamlining it, and experimenting with various algorithms. Ultimately, the SVM model excelled in accurately predicting the success of marketing interactions.

## Reading the Data

We'll load the dataset into our environment using pandas to prepare it for analysis.

## Understanding the Features

This dataset contains information from a marketing campaign by a Portuguese bank to encourage customers to invest in term deposits. Understanding the features in the Portuguese Bank Dataset is vital for tasks like data analysis, making predictive models, and figuring out what influences clients to subscribe to term deposits after the campaign.

 * It's important to check for missing or unknown values in the features.
 * We need to look at how the target variable is spread in the dataset.
 * Also, it's crucial to understand what types of data are in the columns and how unique the categorical features are.

## Understanding the Task

To develop a predictive model using data mining techniques that can determine the likelihood of a customer subscribing to a long-term deposit product as part of a direct marketing campaign by the bank..

## Feature Engineering

In this section, we explore various features, assess their value distributions, identify outliers, evaluate their importance regarding the target variable, and determine if any features can be omitted from the analysis. We rely on visual aids like Heatmaps, histograms, and bar charts to perform these tasks.

For simplifying the training process, we eliminated features that didn't contribute to the target prediction. Additionally, we converted categorical columns into numeric format using pandas' "get_dummies" function.

## Train/Test Split

Dividing the data into training and testing sets to evaluate model performance.

## Baseline Model

A baseline model is a fundamental component in classification tasks as it provides a performance benchmark for advanced models. It serves to identify data issues, understand the problem's inherent complexity, and efficiently allocate resources. When a baseline model performs well, it signifies that the problem may not require complex algorithms, while poor performance indicates a need for more advanced modeling techniques or enhanced feature engineering. Additionally, a baseline model simplifies communication of results to stakeholders, making it a crucial step in the machine learning workflow.

For this dataset, baseline model used is LogisticRegression and baseline accuracy is around 88.865

## Simple Model 

The analysis began by applying a straightforward Logistic Regression model to gain insights into both the model's behavior and the dataset itself. The examination revealed a class imbalance issue in the target variable. A classification report and confusion matrix were generated, shedding light on the disparities in model performance. Notably, for the '0' class of the target, precision, recall, and F1-score all exceeded 90%, indicating strong predictive accuracy. Conversely, for the '1' class, the metrics showed lower values, with precision at 71%, recall at 22%, and an F1-score of just 33%.

## Model Comparisons with default settings

Comparing the performance of K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.

**Logistic Regression**: This model achieved a test accuracy of approximately 89.70%, with a training time of about 10.91 seconds. The training accuracy was also high at 89.87%. Notably, it outperformed the baseline accuracy of approximately 88.87%.

**K-Nearest Neighbors (KNN)**: KNN demonstrated a test accuracy of around 88.86%, with a fast training time of approximately 0.02 seconds. The training accuracy was slightly higher at 91.34%.

**Decision Tree**: The Decision Tree model, while achieving a high training accuracy of approximately 99.50%, had a lower test accuracy of around 83.99%. It should be noted that this test accuracy was lower than the baseline accuracy.

**Support Vector Machine (SVM)**: Similar to Logistic Regression, the SVM model achieved a test accuracy of roughly 89.70%. It had a training time of approximately 11.86 seconds, and the training accuracy matched the test accuracy at 89.87%. The SVM model also outperformed the baseline accuracy.

Each model's test accuracy performed relative to the baseline accuracy of approximately 88.87%.

## Improving the Model and the comparisons

**Logistic Regression:**
- Best Hyperparameters: {'C': 1, 'max_iter': 500, 'penalty': 'l2', 'solver': 'lbfgs'}
- Best Score: 0.90098
- Top Features: 'month_may' (0.5548), 'cons.price.idx' (0.3670), 'contact_cellular' (0.2637), 'contact_telephone' (0.2610), 'poutcome_failure' (0.2223), 'previous' (0.2217)

**K-Nearest Neighbors (KNN):**
- Best Hyperparameters: {'n_neighbors': 14, 'weights': 'uniform'}
- Best Score: 0.89672
- Top Features: 'nr.employed' (0.0241), 'pdays' (0.0104), 'cons.conf.idx' (0.0049), 'age' (0.0013), 'euribor3m' (0.0008), 'previous' (0.0008)

**Support Vector Machine (SVM):**
- Best Hyperparameters: {'C': 0.1, 'kernel': 'rbf'}
- Best Score: 0.89867

**Decision Tree:**
- Best Hyperparameters: {'criterion': 'entropy', 'max_depth': 5, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
- Best Score: 0.90199
- Top Features: 'nr.employed' (0.6749), 'cons.conf.idx' (0.1159), 'poutcome_success' (0.0619), 'euribor3m' (0.0532), 'month_oct' (0.0318), 'pdays' (0.0198)

Decision Tree model stands out with the highest accuracy and interestingly, the 'nr.employed' and 'cons.conf.idx' features are the most influential. The Logistic Regression model also performs well and is strongly influenced by the 'month_may' and 'cons.price.idx' features. KNN, while achieving good accuracy, has feature importance primarily driven by 'nr.employed' and 'pdays'. The SVM model, with relatively fewer hyperparameters, provides a competitive accuracy score.

## Recommendations

In this case dataset is imbalance, so my main recommendations are around that - 
To improve the accuracy of classification models in the presence of data imbalance, it is advisable to employ a combination of strategies. Resampling techniques, such as oversampling and undersampling, can be effective in balancing class distributions. Using alternative evaluation metrics like precision, recall, and the F1-score provides a more comprehensive view of model performance on imbalanced datasets.

Leveraging anomaly detection or collecting more data for the minority class, where possible, can be beneficial. Feature engineering, cross-validation, threshold adjustments, and domain knowledge also play vital roles in mitigating data imbalance challenges. It's crucial to experiment with these recommendations to determine the most suitable approach based on the specific dataset and the objectives of the analysis.

## Data Source
- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

## References
- Article accompanying the dataset: [Link to Article](CRISP-DM-BANK.pdf)