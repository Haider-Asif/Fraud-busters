# Data Spec
This is where you will be describing your data spec comprehensively. Please refer to the handout for a couple of examples of good data specs.

You will also need to provide a sample of your data in this directory. Please delete the example `sample.db` and replace it with your own data sample. ***Your sample does not necessarily have to be in the `.db` format; feel free to use `.json`, `.csv`, or any other data format that you are most comfortable with***.


1) Where the transactions data comes from:
The data is taken from a kaggle notebook, https://www.kaggle.com/mlg-ulb/creditcardfraud?select=creditcard.csv, with a usability rating of 8.5/10, and ~7500 upvotes. The data contains transactions made by credit cards in September 2013 by european cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. 

2) Format of the transactions as in the SQL data file:
The SQL file that we created is a little different from the data itself in the sense that it has an extra column id which is an auto incremented integer and is used for primary key purposes. The other columns of the only table in our database include:

  - Delta_T (Integer) -> This is the difference between the time between the occurrence of the transaction and the time of the first transaction. This field cannot be null because we make sure that the data is cleaned and hence there are no rows with incomplete data or in this case without a relative time stamp.

  - V1-V28 (Float) -> These 28 fields (principal components) represent the different factors that the credit card fraud could be impacted by and their different values as determined after performing principal component analysis on them. The actual names of the variables have not been included in the dataset to preserve anonymity, and reduce the bias introduced in the prediction model. These attributes can also not be null because of the data cleaning performed.

  - Amount (Float) -> This represents the amount of the transaction in dollars. It is not rounded to a nearest integer hence is represented as a float, and cannot be null because of the data cleaning process.

  - Class (Integer) -> This represents the classification of the transaction. This can be an integer 0 or 1, 1 representing a fraudulent transaction, and 0 otherwise. This field is the one that is going to be the label of our data for training and testing purposes, therefore it also cannot be null. 

3) A link to the data, as a sqlite3 database file is here: https://drive.google.com/file/d/1n0wA0D234M8uwOGUSvDG0UeGx1iNty3r/view?usp=sharing 
4) A sample of the data can be found on the link below: https://drive.google.com/file/d/1yXyeTNoltszHPQAMb0d3YCKJm7_jhoOZ/view?usp=sharing 
