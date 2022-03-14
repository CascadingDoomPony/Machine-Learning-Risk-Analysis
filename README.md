# Machine-Learning-Risk-Analysis

This program is a machine learning exercise using various sampling and ensemble learning models to perform risk analysis on sample loan data.

The analysis is split into two Jupyter Notebook files, both of which use different datasets for their learning and testing. credit_risk_resampling.ipynb tries out different kinds of sampling methods and uses the logistic regression model to test them. credit_risk_ensemble.ipynb tests an expanded dataset using two imbalanced learning ensemble learning models.

The data we use is split between training and testing data using sklearn's train_test_split method, then scaled using sklearn's StandardScaler 

We are using the sklearn and imblearn python packages to do our processing and modelling. The metrics we use include a balanced accuracy score, confusion matrix, and classification report. You can see all these metrics in the notebook files

## Resampling methods

We ran a logistics regression model on a dataset that has been resampled either randomly oversampled, using SMOTE, Cluster Centroids, or with SMOTEEN. We also fit the model to an unsampled dataset as a benchmark

The first thing we should note is that the difference between the performance on all of these models is very small, and they all performed very well. For example, the model using an unsampled dataset got a 98.8% accuracy score, and all the other models performed at above 99%. This could be because there is a clear difference between low risk and high risk datapoints in our dataset, and it is inherently easy to recognise the difference between the two. Some evidence that points to this is the SMOTEEN oversampler. The high risk dataset is 54854 big, and for low risk it is 55912, which indicates that in the dataset of synthetically generated data points in the high risk dataset, only 1058 data points were removed for being too close to the low risk data points by edited nearest neighbours (ENN), a very small amount considering the size of our dataset.

Overall it appears that the random oversampler and SMOTEEN sampler helped the model the most, slightly outperforming other methods with an accuracy score of 99.34%. Although recall is almost indistinguishable between the sampling methods, the Cluster Centroids undersampler performed slightly worse than all others.

## Ensemble learning

The dataset we use for the ensemble learners contains many more fields than with the previous section. We ran two ensemble learner models, a Balanced Random Forest Classifier and an Easy Ensemble Classifier, both from the imblearn package and both run with 100 n-estimators. The Easy Ensemble classifier performs slightly better with 72.8% balanced accuracy compared to 70.2%, but the Forest Classifier has a better recall with 0.81 against 0.76

