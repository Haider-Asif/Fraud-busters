# Tech Report
This is where you can type out your tech report.

### Where is the data from? ###
The data is taken from a kaggle notebook, https://www.kaggle.com/mlg-ulb/creditcardfraud?select=creditcard.csv, with a usability rating of 8.5/10, and ~7500 upvotes, which leads us to believe that the source is reputable, and verified by the educational community. The data contains transactions made by credit cards in September 2013 by european cardholders.

This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. 

Because the data is based on transactions that occurred in only two days, it is relatively small compared to the total number of transactions. The data can also exhibit some sampling bias because we do not know which two days in september the data transactions were collected on, because days at the end of the month when people are low on cash can result in higher number of transactions, and possibly higher number of fraudulent transactions
The dataset is very skewed because the number of transactions that are fraudulent are really less only 0.172% in the real world as compared to the total number of transactions. This is why we plan on measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC).

### How clean is the data? Does this data contain what you need in order to complete the project you proposed to do? (Each team will have to go about answering this question differently, but use the following questions as a guide. ###

There are 284,807 rows of data (transactions) in total. There are only 492 fraudulent transactions, classified as 1 in the class field of the data, and 284,3125 non-fraudulent transactions classified as 0 in the class field of the data. The data is highly imbalanced, but we believe that it should be enough to conduct analysis on. The reason being that this models a real life scenario where fraudulent transactions are much less than the total number of credit card transactions, and we plan on measuring the accuracy of the model using the Area Under the Precision-Recall Curve (AUPRC) which allows us to analyze models that train over imbalanced data because it takes into account true positives, false negatives, false positives but does not take into account true negatives because of which we can get an accurate idea of how the model performs even with the imbalanced data.       
There is no missing data, or duplicates present in the data, as per the data specification from the kaggle notebook. We do not need to throw any data away, and did not have any data type issues because the creators of this dataset already cleaned the data beforehand for model training purposes. Finally the data distribution is skewed class wise as mentioned above, with a very large number of non-fraudulent transactions compared to the fraudulent transactions.

### Summarize any challenges or observations you have made since collecting your data. Then, discuss your next steps and how your data collection has impacted the type of analysis you will perform. (approximately 3-5 sentences) ###

The main challenge that exists in our dataset is the anonymized features, as the dataset includes 29 features, and two known features (relevant time of transaction and amount of transaction). This would pose also a challenge in deducing conclusions in relation to historical and social context, and analyzing the conclusions in relation to the feature. At the same time, the anonymization of the data can ensure personal bias does not factor into conclusions drawn. This obviously favors quantitative data and gives it the power to draw the conclusions, and would restrict our ability to blend in qualitative research to the conclusions drawn by the anonymized quantitative features. Another challenge is the imbalance between fraud and non-fraud transactions, as the non-fraudulent transactions make up only 0.172% of the transactions. We plan on overcoming this imbalance ratio by using the Area Under the Precision-Recall Curve (AUPRC).

For more ways to organize your report, check this markdown cheatsheet: https://github.com/tchapi/markdown-cheatsheet/blob/master/README.md

We ***highly encourage you to use markdown syntax to organize and clean your reports*** since it will make it a lot more comprehenisble for the TA grading your project.
