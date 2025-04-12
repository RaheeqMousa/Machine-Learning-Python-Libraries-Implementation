import pandas as pand
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as seab
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score,mean_absolute_error, classification_report,confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


file_url='Customer Churn.csv'
read_data=pand.read_csv(file_url)

empty=read_data.isnull().sum()
print(empty)


#Task: print the Full Summary of the Dataframe
print(read_data.info()) 

# Distribution of 'Churn' column
read_data['Churn'].value_counts().plot(kind='bar')
plt.title('Distribution of the attribute Churn')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()




#Task: for each age group, draw a histogram detailing the amount of churn in each sub-group.

#.groupby(['Age Group', 'Churn']) this will group the data based on the "Age Group" and "Churn" columns.
#.size() counts the number of rows in each group
#
churn_counts = read_data.groupby(['Age Group', 'Churn']).size().unstack()
churn_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xlabel('Age Group')
plt.ylabel('Number of Customers')
plt.title('Churn Distribution by Age Group')
plt.show()


min_age_for_age_group=read_data.groupby('Age Group')['Age'].min()
max_age_for_age_group=read_data.groupby('Age Group')['Age'].max()
print("minimum age for age group")
print(min_age_for_age_group)
print("maximum age for age group")
print(max_age_for_age_group)


#Task: For each charge amount, draw a histogram detailing the amount of churn in each sub-group.

churn_by_charge_amount = read_data.groupby(['Charge Amount', 'Churn']).size().unstack()
churn_by_charge_amount.plot(kind='bar', stacked=True, color=['pink', 'blue'], figsize=(10, 6))
plt.title('Churn Distribution by Charge Amount')
plt.xlabel('Charge amount')
plt.ylabel('Number of customers')
plt.show()


#Task:Show the details of the charge amount of customers.
charge_amount_details=read_data['Charge Amount'].describe()
print("Charge Amount Details")
print(charge_amount_details)

#Task: Visualise the correlation between all features and explain them in your own words.

correlation_data = read_data

#correlation matrix is made for a numeric distribution so i should display the categorical data as numeric data
correlation_data['Churn'] = correlation_data['Churn'].map({'no': 0, 'yes': 1})
correlation_data['Status'] = correlation_data['Status'].map({'not-active': 0, 'active': 1})
correlation_data['Plan'] = correlation_data['Plan'].map({'pre-paid': 0, 'post-paid': 1})
correlation_data['Complains'] = correlation_data['Complains'].map({'no': 0, 'yes': 1})

correlation_matrix = correlation_data.corr()
seab.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Features Correlations")
plt.show()


#Task: split data set into 70% training set and 30% test set

#x is the kept on features
x = read_data.drop(['Churn'],axis=1)
y=read_data['Churn']
X_train, X_test, y_train, y_test= train_test_split(x,y,test_size=0.3, random_state=0)
print('training set')
print(X_train.shape)
print('testing set')
print(X_test.shape)


#Regression tasks
#1.Apply linear regression to learn the attribute “Customer Value” using all independent attributes (call this model LRM1).

LRM1 = LinearRegression()

LRM1.fit(X_train, y_train)

y_predicted = LRM1.predict(X_test)

mse = mean_squared_error(y_test, y_predicted)
mean_absolute_error_LRM1 = mean_absolute_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

print("\n")
print("Linear Regression \"First model\" measurements")
print("Mean Squared Error (LRM1): ",mse)
print("Mean Absolute Error (LRM1): ",mean_absolute_error_LRM1)
print("R-squared (LRM1): ",r2)

#2.Apply linear regression using the set of the 2 most important features (from your point of view); and explain why did you use these 2 attributes (call this model LRM2).
LRM2=LinearRegression()

x_for_LRM2= read_data[['Distinct Called Numbers', 'Age Group']]
y_for_LRM2= read_data['Customer Value']

x_train_lrm2,x_test_lrm2,y_train_lrm2,y_test_lrm2= train_test_split(x_for_LRM2,y_for_LRM2,test_size=0.3, random_state=0)

LRM2.fit(x_train_lrm2,y_train_lrm2)
predicted_y_lrm2=LRM2.predict(x_test_lrm2)

mean_absolute_error_lrm2 = mean_absolute_error(y_test_lrm2, predicted_y_lrm2)
mean_square_error_lrm2 = mean_squared_error(y_test_lrm2, predicted_y_lrm2)
r2_lrm2= r2_score(y_test_lrm2, predicted_y_lrm2)

print("\n")
print("Linear Regression \"Second model\" measurements")
print("Mean Squared Error= ", mean_square_error_lrm2)
print("Mean Absolute Error= ", mean_absolute_error_lrm2)
print("R squared= ", r2_lrm2)

#3. Apply linear regression using the set of the most important features (based on the correlation coefficient matrix) and explain why did you use these attributes (call this model LRM3).


LRM3_a = read_data[['Freq. of use', 'Freq. of SMS', 
               'Distinct Called Numbers', 'Status']]

# the dependent variables are customer value, and churn
LRM3_b=read_data['Customer Value']
lrm3_x_train, lrm3_X_test, lrm3_y_train, lrm3_y_test= train_test_split(LRM3_a,LRM3_b,test_size=0.3, random_state=0)

LRM3=LinearRegression()
LRM3.fit(lrm3_x_train,lrm3_y_train)
predicted_y3=LRM3.predict(lrm3_X_test)#the prediction is made on the test data   

mean_square_error3 = mean_squared_error(lrm3_y_test, predicted_y3) #tells me how far the predicted values of "Customer Value" are from the actual values
mean_absolute_error_lrm13 = mean_absolute_error(lrm3_y_test, predicted_y3)
accuracy_lrm3 = LRM3.score(lrm3_X_test, lrm3_y_test)
r2_lrm3= r2_score(lrm3_y_test, predicted_y3)

print("\n")
print("Linear Regression \"Third model\" measurements")
print("Mean Absolute Error= ", mean_absolute_error_lrm13)
print("Mean Squared Error= ", mean_square_error3)
print("accuracy= ", accuracy_lrm3)
print("R squared= ", r2_lrm3)

# print("Coefficients:", LRM3.coef_)
# print("Intercept:", LRM3.intercept_)

#4. Compare the performance of these models using adequate performance metrics

performance= pand.DataFrame({
    'Model':['LRM1' ,'LRM2', 'LRM3'],
    'MSE':[mse, mean_square_error_lrm2, mean_square_error3],
    'MAE':[mean_absolute_error_LRM1,mean_absolute_error_lrm2,mean_absolute_error_lrm13],
    'R^2':[r2,r2_lrm2, r2_lrm3]
})

print(performance)


#Classification tasks:
#1. Run k-Nearest Neighbours classifier to predict churn of customers (the “Churn” feature) using the test set

#Before training kNN model, we must scale the data before because it is a
#distance algorithm based since it calculates the distance between data points from
#training set and the inputs from the test set to classify them). So if the features have
#different scales, features with larger numerical ranges will control the distance
#calculation, leading to biased predictions.

scaler = StandardScaler() #standardize features by removing the mean and scaling to unit varianc
X_train_scaler = scaler.fit_transform(X_train)
x_test_scaler= scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaler, y_train)

predicted_y_KNN=knn.predict(x_test_scaler)

accuracy_score_knn=accuracy_score(y_test, predicted_y_KNN)
confusion_matrix_knn=confusion_matrix(y_test,predicted_y_KNN)
classification_report_knn=classification_report(y_test,predicted_y_KNN)
y_probabilty_knn = knn.predict_proba(x_test_scaler)[:, 1] # probability of churn
roc_auc_knn = roc_auc_score(y_test, y_probabilty_knn)

print("\n")
print("K-Nearest Neighbours classifier measurements")
print("Accuracy Score= ", accuracy_score_knn)
print("Confusion Matrix:\n",confusion_matrix_knn)
print("classifiction report = \n", classification_report_knn)
print("ROC-AUC Score:", roc_auc_knn)

#2. Run Naive Bayes classifier to predict churn of customers (the “Churn” feature) using the test set

gauss_naive_bayes = GaussianNB()
gauss_naive_bayes.fit(X_train_scaler, y_train)

predicted_y_gnb=gauss_naive_bayes.predict(x_test_scaler)

accuracy_score_gnb=accuracy_score(y_test, predicted_y_gnb)
confusion_matrix_gnb=confusion_matrix(y_test, predicted_y_gnb)
classification_report_gnb=classification_report(y_test, predicted_y_gnb)

y_probabilty_naive = gauss_naive_bayes.predict_proba(x_test_scaler)[:, 1] # probability of churn
roc_auc_naive = roc_auc_score(y_test, y_probabilty_naive)

print("\n")
print("Naive Bayes classifie measurements")
print("Accuracy Score= ", accuracy_score_gnb)
print("Confusion Matrix:\n",confusion_matrix_gnb)
print("classifiction report = \n", classification_report_gnb)
print("ROC-AUC Score:", roc_auc_naive)

#3. Run Decision Tree classifier to predict churn of customers (the “Churn” feature) using the test set

decision_tree = DecisionTreeClassifier(max_depth=5,random_state=1)
decision_tree.fit(X_train,y_train)

predicted_y_DT=decision_tree.predict(X_test)
y_probabilty_DT=decision_tree.predict_proba(X_test)[:,1]

roc_auc_dt = roc_auc_score(y_test,y_probabilty_DT)
train_accuracy_score_dt = accuracy_score(y_train, decision_tree.predict(X_train))
test_accuracy_score_dt=accuracy_score(y_test, predicted_y_DT)
confusion_matrix_dt=confusion_matrix(y_test,predicted_y_DT)
classification_report_dt=classification_report(y_test, predicted_y_DT)

print("\n")
print("Decision Tree classifier Results")
print("Accuracy Score on train set", train_accuracy_score_dt)
print("Accuracy Score on test set= ", test_accuracy_score_dt)
print("Confusion Matrix:\n",confusion_matrix_dt)
print("classifiction report = \n", classification_report_dt)
print("ROC-AUC Score:", roc_auc_dt)


#4. Compare the performance of Logistic regression, Naive Bayes, and kNN classifiers in an appropriate results section. Compare the classification performance of the generated classification models
#and make sure to use the appropriate performance metrics. You should include at least the ROC/AUC score and the Confusion Matrix.
#Report the results in an appropriate table and explain in your own words why one model outperforms the other. 


#build the logistic regression
logistic_regression = LogisticRegression(random_state=0)
logistic_regression.fit(X_train_scaler, y_train)

predicted_y_logistic=logistic_regression.predict(x_test_scaler)
accuracy_score_logistic=accuracy_score(y_test, predicted_y_logistic)
confusion_matrix_logistic=confusion_matrix(y_test, predicted_y_logistic)
classification_report_logistic=classification_report(y_test, predicted_y_logistic)
y_probabilty_logistic = logistic_regression.predict_proba(x_test_scaler)[:, 1] # probability of churn
roc_auc_logistic = roc_auc_score(y_test, y_probabilty_logistic)

print("\n")
print("Logistic Regression measurements")
print("Accuracy Score= ", accuracy_score_logistic)
print("Confusion Matrix:\n",confusion_matrix_logistic)
print("classifiction report = \n", classification_report_logistic)
print("ROC-AUC Score:", roc_auc_logistic)
