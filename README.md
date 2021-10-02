# Credit Risk Analysis📋  📄  📃  📑  
*Supervised Machine Learning* 
## Project Overview 
For this project we are using several models of supervised Machine Learning on credit 
loan data in order to predict credit risk.    We used Python Sickit-learn library and several machine
learnings models (Supervised Machine Learning Models - Logistic Regression, Random Forest, AdaBoost Classfier,
Cluster Centroids, Oversampling & Undersampling).  We will use several models of supervised machine learning on credit loan data in order 
top predict credit risk.  Credit risk is an inherently unbalanced classification problem, and risky loans outnumber good loans.  We will 
use Python and Scikit-learn libraries and several machine learning models to compare the strengths and weeknesses of machine 
learning models and determine how well a model classifies and predicts data.   This technique and information will minimize risk for 
loan agent.  

#### Resources: 
-  Data sources:  **LoanStats_2019.csv** (very large data file - zipped and included in this repository)
-  Sofware: **Jupyter Notebook**
-  Languages: **Python**
-  Libraries: **SKLearn/Scikit-learn**, **Pandas**, **Matplotlib**
-  Environment: **Python 3.7**
 
## Executive Brief  
In this analysis, we used six different algorithms of **supervised machine learning**.  The First 
four algorithms are based on **resampling techniques** and are designed to deal with **class 
imbalance**.  
After the data is resampled, **Logistic Regression** is used to predict the outcome.   Logistic 
regression predicts binary outcomes (A) (in one class or another). 
The last two models are from ensemble learning group.  The concept of ensemble learning is the 
process of combining multiple models, like decision tree algorithms, to help improve the accuracy
 and robustness, as well as decrease variance of the model, and therefore increase the overall 
performance of the model(F).   

##  Deliverable 1, (A-C) - Three Algorithms for Prediction including: 
#### A) Naive Random Oversampling
#### B) SMOTE Oversampling and Logistic Regression, &
#### C) Cluster Centroids Undersampling and Logistic Regression

### Algorithm A. Naive Random Oversampling:
Naive Random Oversampling and Logistic Regression 
In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority
classes are balanced(B).   
-   **Accuracy score**:    65%
-   **Precision - for high risk**:  0.01
-   **Precision - for low risk**: 1.00

-   **Recall - for high risk**:  0.72
-   **Recall - for low risk**:  0.59

<p align="center">
  <img width="550" height="700" src="https://github.com/mjrotter4445/Credit_Analysis/blob/main/Graphics/Fig1.jpg">
</p>
<p align="center">
Figure 1-Results for Naive Random Oversampling
</p>
 
### Algorithm B. SMOTE Oversampling and Logistic Regression:
The **synthetic minority oversampling technique (SMOTE)** is another oversampling approach where the minority class increased.  Unlike other oversampling methods, SMOTE 
interpolated new instances, that is, for an instance from the minority class, a number of its closest neighbors is chosen.  Based on the values, these neighbors, new values are 
created(C & E) 

-   **Accuracy score**:    64% 
-   **Precision - for high risk**:  0.01
-   **Precision - for low risk**: 1.00

-   **Recall - for high risk**:  0.60
-   **Recall - for low risk**:  0.69

<p align="center">
  <img width="550" height="700" src="https://github.com/mjrotter4445/Credit_Analysis/blob/main/Graphics/Fig2.jpg">
</p>
<p align="center">
Figure 2-Results for SMOTE Oversampling
</p>

### Algorithm C. Cluster Centroids Undersampling and Logistic Regression:
Undersampling takes the opposite approach of oversampling.  Instead of
increasing the number of the miniority class, the size of the majority class is decreased (4).      

-   **Accuracy score**:    54%
-   **Precision - for high risk**:  0.01
-   **Precision - for low risk**: 1.00

-   **Recall - for high risk**:  0.69
-   **Recall - for low risk**:  0.40

<p align="center">
  <img width="550" height="700" src="https://github.com/mjrotter4445/Credit_Analysis/blob/main/Graphics/Fig3.jpg">
</p>
<p align="center">
Figure 3-Results for Cluster Centroids Undersampling
</p>

## Deliverable 2 - Algorithm D SMOTEENN (Combination of Over and Under Sampling) and Logistic Regression:
SMOTEENN is an approach to resampling that combines aspects of both oversampling and undersampling - oversample the miniority class with SMOTE(4).       

-   **Accuracy score**:    66%
-   **Precision - for high risk**:  0.01
-   **Precision - for low risk**: 1.00

-   **Recall - for high risk**:  0.76
-   **Recall - for low risk**:  0.57

<p align="center">
  <img width="550" height="700" src="https://github.com/mjrotter4445/Credit_Analysis/blob/main/Graphics/Fig4.jpg">
</p>
<p align="center">
Figure 4-Results for SMOTTEENN Model  
</p>

##  Deliverable 3 - (A & B) 
### Deliverable 3A -  Algorithm E Balanced Random Forest Classifier
Instead of having a dingle, complex tree like the ones created by decision trees, a random forest algorithm will sample the data and build several smaller, simpler decision 
trees.  Each tree is simpler because it is built from a random subset of features(H).         

-   **Accuracy score**:    79%
-   **Precision - for high risk**:  0.03
-   **Precision - for low risk**: 1.00

-   **Recall - for high risk**:  0.70
-   **Recall - for low risk**:  0.87

<p align="center">
  <img width="550" height="700" src="https://github.com/mjrotter4445/Credit_Analysis/blob/main/Graphics/Fig5.jpg">
</p>
<p align="center">
Figure 5-Results for Random Forest Classifier Model 
</p>

### Deliverable 3B - Algorithm F Easy Ensemble AdaBoost Classifier
In **AdaBoost Classifier**, a model is trained then evaluated(I).             

-   **Accuracy score**:    93%
-   **Precision - for high risk**:  0.09
-   **Precision - for low risk**: 1.00

-   **Recall - for high risk**:  0.92
-   **Recall - for low risk**:  0.94

<p align="center">
  <img width="550" height="700" src="https://github.com/mjrotter4445/Credit_Analysis/blob/main/Graphics/Fig6.jpg">
</p>
<p align="center">
Figure 6-Easy Ensemble AdaBoost Classifier Model  
</p>

##  Summary
From the results section above we can see how different ML models work on the 
same data.  I would like to start the interpretation of the results with a 
brief explanation of the Measures we evaluate in the outcomes.    

**ACCURACY SCORE** tells us what percentage of predictions themodel gets it right. However, it is not enough just to see that results, especially with unbalanced data.  
*Equation: 
accuracy score = number of correct prediction/total number of predictions.*  

**PRECISION** is the measure of how reliable a positive classification is.  A low precision is indicative of a large number of false positives.  *Equation: Precision = TP/(TP + 
FP)*

**RECALL** is the ability of the classifier to find all the positive samples.  A low recall is indicative of a large number of false negatives *Equation:Recall = TP/(TP+FN)*   

**FI SCORE** is weighted average of the true positive rate (recall) and precision, where the best score is 1.0. *Equation: F1 score = 
2(Precision x Sensititivity)/(Precision + Sensitivity*)
The F1 Score equation is:   2*((precision*recall)/(precision+recall)). It is also called the F Score or the F Measure. Put another way, 
the F1 score conveys the balance between the precision and the recall. The F1 for the All No Recurrence model is 2*((0*0)/0+0) or 0.  

**Results Summary**(H)

**First 4 models (Figures 1-4 above) - resampling and logistic regression**
From the results above we can see that first four models don't do well based off the **accuracy scores**.  Those scores are 65%, 64%, and 54% for Naive Random Oversampling, 
SMOTE Oversampling, Cluster Centroids Undersampling and SMOTEENN model respectively, meaning the models were accurate roughly a a little more than 1/2 (or 50%) of the time.  

**Precision** for all four models is 0.01 for high risk loans and 1.00 for low risk loans.  **Low precision scores** for high risk loans 
is due to large number of false 
positives, meaning that too many of low risk loans were mrked as high risk loans.   High score for low risk loans indicate that nearly 
all low risk scores were marked correctly; 
however, lower **recall score** (.072 for naive Naive Random Oversampling and Logistic Regression, for example) indicates that there wer 
quite a few low risks loans that were 
market as high risk, when they actually were not.   Actual high risk loans have slightly better scores on recall (.069 for naive Naive 
Random Oversampling and Logistic Regression, for example) meaning that there weren't as many false negatives or not too many high risk 
loans were marked as low risk loans.   

**Last 2 models (Figures 5 & 6 above) - Ensemble models**
The last two models worked better.    Their accuracy scores are 79% and 93% for Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier respectively.   **Recall 
scores** for both model and both - low and high risk scores and precision for low risk were high, meaning very good accuracy. 
**Precision** for high risk loans in both models 
were not high at 0.03 and 0.09 for Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier respectively, indicating that 
there were **large number of false 
positives**, meaning that large number of low risk loans were marked as high risk.     

**Recommendation on the model**
Since the first three models didn't do well on the test, I would use them in real-world testing without further fine-tuning, for example 
train model on larger dataset, or look through the columns that were used for training the model.   The other 2 models showed better results, yet I would use them with caution, 
since they might be prone to overfitting. If that occurs and we don't get desired results when working with new data set, we can do some 
further fine-tuning (pruning) to avoid the overfitting.(H)   

**References**:   
Module 17:  
(A) 17.3.1 Logistic Regression
(B) 17.4.2 Create a confusion matrix
(C) 17.10.1 Oversampling
(D) 17.7.3 Make Predictions and Evaluate Results
(E) 17.10.2 Undersampling
(F) 17.10.3 Combination Sampling With SMOTEENN
(G) 17.8.1 Overview of Ensemble Learning 
(H) 17.8.3 Fit the Model, Make Predictions, and Evaluate Results
(I) 17.9.2 Boosting 
:) 
