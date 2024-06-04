import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
#import  TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer

#count_vectorizer = CountVectorizer()

df = pd.read_csv('E:\final project\Phishing Website\Phishing Website\phishing.csv')


#print("Total Dataset:", len(df))

print(df.isna().sum())
X = df.Domain # X_feature
y = df.Label # y_label

print(X)
print(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

count_vectorizer.fit(X_train)
X_train_cv = count_vectorizer.transform(X_train)
tfidf_transformer.fit(X_train_cv)
X_train_tfidf = tfidf_transformer.transform(X_train_cv)

X_test_cv = count_vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_cv)




import xgboost as xgb
from sklearn.metrics import accuracy_score
# Train an XGBoost model
#xgb_model = xgb.XGBClassifier()
#xgb_model.fit(X_train, y_train)
xgb_classifier = xgb.XGBClassifier(learning_rate=1.0, max_depth=7, n_estimators=300)
xgb_classifier.fit(X_train_tfidf, y_train)

y_pred_xgb = xgb_classifier.predict(X_test_tfidf)


accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", accuracy_xgb)



import pickle

# Save the trained Random Forest classifier to a file
with open('XGBoost.pkl', 'wb') as f:
    pickle.dump(xgb_classifier, f)





# Define XGBoost classifier with specified parameters




from sklearn.ensemble import RandomForestClassifier


rf_classifier = RandomForestClassifier()

# Fit the classifier to the training data
rf_classifier.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred_rf = rf_classifier.predict(X_test_tfidf)

# Evaluate the performance of the classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)


'''
import pickle

# Save the trained Random Forest classifier to a file
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)
'''



# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier


dt_classifier = DecisionTreeClassifier( max_depth=3 )


dt_classifier.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred_dt = dt_classifier.predict(X_test_tfidf)

# Evaluate the performance of the classifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)


import pickle

# Save the trained Random Forest classifier to a file
with open('decisiontree.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)


from sklearn.neural_network import MLPClassifier

# Instantiate the MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=10, alpha=0.0001,
                     solver='adam', verbose=10,  random_state=42, tol=0.0001)

# Fit the classifier to the training data
mlp_classifier.fit(X_train_tfidf, y_train)


y_pred_mlp = mlp_classifier.predict(X_test_tfidf)


accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("MLP Accuracy:", accuracy_mlp)



from sklearn.neighbors import KNeighborsClassifier


knn_classifier = KNeighborsClassifier(n_neighbors=5)


knn_classifier.fit(X_train_tfidf, y_train)


y_pred_knn = knn_classifier.predict(X_test_tfidf)


accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)



from sklearn.naive_bayes import MultinomialNB

# Instantiate the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred_nb = nb_classifier.predict(X_test_tfidf)

# Evaluate the performance of the classifier
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Accuracy:", accuracy_nb)



from sklearn.ensemble import AdaBoostClassifier


adaboost_classifier = AdaBoostClassifier()


adaboost_classifier.fit(X_train_tfidf, y_train)


y_pred_adaboost = adaboost_classifier.predict(X_test_tfidf)


accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
print("AdaBoost Accuracy:", accuracy_adaboost)



import matplotlib.pyplot as plt


classifiers = ['XGBoost', 'Random Forest', 'Decision Tree', 'MLP', 'KNN', 'Naive Bayes', 'AdaBoost']
accuracies = [accuracy_xgb, accuracy_rf, accuracy_dt, accuracy_mlp, accuracy_knn, accuracy_nb, accuracy_adaboost]

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.barh(classifiers, accuracies, color='skyblue')
plt.xlabel('Accuracy')
plt.title('Accuracy of Different Classifiers')
plt.xlim(0, 1)  # Set x-axis limits between 0 and 1
plt.show()

