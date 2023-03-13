import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support,roc_curve,roc_auc_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE


def main():

	
	st.title("Upload any file type and see its contents as a Dataframe")

	file = st.file_uploader("Upload your file") #, type=["csv", "txt", "xlsx", "xls"])

	if file is not None:
		try:	
			df = pd.read_csv(file)

			st.dataframe(df)

		except Exception as e:
			st.error("Could not load file as a Dataframe.")
			st.exception(e)

	st.title("Tune the file reader parameters")

	input_sep = st.text_input("Enter your data's separator, or leave empty for automatic detection:")

	input_header = st.number_input("Enter the row number of the header, or at zero for automatic detection:",step=1, min_value=0)

	input_column_names = st.text_input("Enter your data's column names if there is no header: (use ',' to separate the names)")

	

	if file is not None:
		file.seek(0)
		try:
			if (input_sep==''):
				sep=","
			else:
				sep = input_sep
			if (input_header==0):
				header="infer"
			else:
				header= input_header

			if(input_column_names==""):
				names = None
			else:
				names = input_column_names.split(",")

			df2 = pd.read_csv(file, sep=sep, header=header, names=names)

			st.write("Current state of the data:")

			st.dataframe(df2)

			st.session_state['df'] = df2
		except Exception as e:
			st.error("Could not load file as a Dataframe.")
			st.exception(e)

	
	

def page2():

	df=st.session_state['df']

	st.title("Your data")


	st.dataframe(df)

	st.header("Choose the Target column:")
	target_col=st.selectbox("",tuple(df.columns),label_visibility="collapsed", help="This is the value that will be predicted by the model you build, make sure it contains no missing values")
	st.write("the target (also referred to as the dependent variable or response variable) is the variable that the model is trying to predict. It is the output or the response that the model is trying to learn to predict based on the input variables.")


	y = df[target_col]
	ordinal_encoder = OrdinalEncoder()
	y = pd.DataFrame(ordinal_encoder.fit_transform(y.values.reshape(-1,1))).reset_index(drop=True)
	X = df.drop([target_col], axis=1)


	
	X2=X.copy()
	del_cols = []


	st.header("Drop some of the columns?")
	del_cols=st.multiselect("",X.columns,label_visibility="collapsed")

	X2 = X.drop(del_cols,axis=1)

	st.header("Training data percentage")
	train_size = st.slider('', 1, 99, 80,label_visibility="collapsed")
	train_size = train_size /100
	st.write("When building a model, it is important to have an independent data set to assess the performance of the model. To achieve this, it is common practice to split the available data set into two subsets: a training set and a validation set.")
	st.write("Typically, the split ratio between the training and validation sets can vary depending on the size of the available dataset. A common practice is to use a 80/20 or 70/30 split, where the larger portion is used for training the model and the smaller portion is used for validating the performance of the model. However, other split ratios can be used depending on the specific requirements of the problem.")
	X_train, X_valid, y_train, y_valid = train_test_split(X2, y, train_size=train_size, test_size=1-train_size,
	                                                            random_state=0)

	##############



	st.title("Dealing with missing values")

	st.header("Special characters as NULL")
	special_char = st.text_input("Enter any special strings that are considered a Null Value (use commas to separate multiple)")
	special_char= special_char.split(",")
	for char in special_char:
		X_train = X_train.replace(char, np.nan)
		X_valid = X_valid.replace(char, np.nan)

	with st.expander(label='Number of Missing Values:', expanded=True):
		st.write(pd.concat([X_train, X_valid], axis=0).isna().sum())

	#radio_missing_values = st.radio("How to deal with the missing values?", ('Drop the rows containing Null values', 'Choose the method for each column'))
	del_cols = []
	# if radio_missing_values == 'Drop the rows containing Null values':
	# 	df2 = df.dropna()
	#else: ##method for each column
	missing_values_cols = X2.columns[X2.isna().any()].tolist()
	st.header("Method to deal with missing values")
	st.write("Missing values in a dataset refer to the absence of a particular value in a certain observation or record. In other words, missing values occur when no data is recorded for a particular attribute of an observation or record.")
	
	st.write("There are various methods to handle missing values, some of which are: \n ")
	st.write("- Dropping columns: This method involves removing the entire column that contains missing values. This method is suitable when the column has a large number of missing values and is not important for the analysis. However, it may result in the loss of relevant information if the column contains important data. ")
	st.write("- Imputation: This method involves filling in the missing values with some value. There are several techniques for imputing missing values, such as:")
	st.write("	a. Mean imputation: This method involves replacing missing values with the mean or median of the available data. This method is simple and effective when the data is normally distributed and the number of missing values is small.")
	st.write("	b. Mode imputation: This method involves replacing missing values with the mode of the available data. This method is suitable for categorical data. ")
	st.write("- Imputation with marking the missing value: This method involves filling in the missing values with some value and marking them as missing. This method allows the machine learning algorithms to differentiate between actual values and imputed values.")
	X_train=X_train.reset_index(drop=True)
	X_valid=X_valid.reset_index(drop=True)
	X_train_no_miss= X_train.copy()
	X_valid_no_miss = X_valid.copy()
	
	for col in missing_values_cols:	
		
		missing_values_method = st.selectbox("For the column '"+col+"':", ('Drop the column', 
			'mean Imputation','mean Imputation + create a new column to indicate missing value'
			,'mode Imputation','mode Imputation + create a new column to indicate missing value'))
		if missing_values_method == 'Drop the column':
			del_cols.append(col)
		
		elif missing_values_method == 'mode Imputation':
			try:
				my_imputer = SimpleImputer(strategy='most_frequent')
				X_train_no_miss[col] = pd.DataFrame(my_imputer.fit_transform(X_train[col].values.reshape(-1,1))).reset_index(drop=True)
				X_valid_no_miss[col] = pd.DataFrame(my_imputer.transform(X_valid[col].values.reshape(-1,1))).reset_index(drop=True)
				
			except Exception as e:
				st.error("Could not perform Imputation.")
				st.exception(e)
				
		elif missing_values_method == 'mean Imputation':				
			try:
				my_imputer = SimpleImputer(strategy='mean')
				X_train_no_miss[col] = pd.DataFrame(my_imputer.fit_transform(X_train[col].values.reshape(-1,1))).reset_index(drop=True)
				X_valid_no_miss[col] = pd.DataFrame(my_imputer.transform(X_valid[col].values.reshape(-1,1))).reset_index(drop=True)
				
			except Exception as e:
				st.error("Could not perform Imputation.")
				st.exception(e)
		elif missing_values_method=='mode Imputation + create a new column to indicate missing value': 
			try:
				my_imputer = SimpleImputer(strategy='most_frequent')
				X_train_no_miss[col + '_was_missing'] = X_train[col].isnull().astype(int)
				X_train_no_miss[col] = pd.DataFrame(my_imputer.fit_transform(X_train[col].values.reshape(-1,1))).reset_index(drop=True)
				X_valid_no_miss[col + '_was_missing'] = X_valid[col].isnull().astype(int)
				X_valid_no_miss[col] = pd.DataFrame(my_imputer.transform(X_valid[col].values.reshape(-1,1))).reset_index(drop=True)
			except Exception as e:
				st.error("Could not perform Imputation.")
				st.exception(e)
		else:						## mean imputation + new col to indicate null
			try:
				my_imputer = SimpleImputer(strategy='mean')
				X_train_no_miss[col + '_was_missing'] = X_train[col].isnull().astype(int)
				X_train_no_miss[col] = pd.DataFrame(my_imputer.fit_transform(X_train[col].values.reshape(-1,1))).reset_index(drop=True)
				X_valid_no_miss[col + '_was_missing'] = X_valid[col].isnull().astype(int)
				X_valid_no_miss[col] = pd.DataFrame(my_imputer.transform(X_valid[col].values.reshape(-1,1))).reset_index(drop=True)
			except Exception as e:
				st.error("Could not perform Imputation.")
				st.exception(e)

	X_valid_no_miss = X_valid_no_miss.drop(del_cols,axis=1)			
	X_train_no_miss = X_train_no_miss.drop(del_cols,axis=1)		
	st.write("The state of your data now:")
	st.dataframe(pd.concat([X_train_no_miss, X_valid_no_miss], axis=0))


	st.title("Dealing with categorical variables")
	st.write("Categorical variables are variables that take a limited number of values or categories, which represent a qualitative or nominal measurement scale. These variables do not have a numerical meaning, but rather describe a characteristic or attribute of the data.")
	cat_cols = dict()
	X_train_cat = X_train_no_miss.copy().reset_index(drop=True)
	X_valid_cat = X_valid_no_miss.copy().reset_index(drop=True)

	cat_cols = st.multiselect("Choose the categorical variables:",list(set(X_valid_cat.columns) | set(X_train_cat.columns)))
	st.write("Categorical variables need to be properly encoded in order to be used in models. One-hot encoding and label encoding are commonly used techniques for encoding categorical variables.")
	X_valid_cat2=X_valid_cat.reset_index(drop=True)
	X_train_cat2=X_train_cat.reset_index(drop=True)
	
	with st.expander(label='Choose the methods to deal with the categorical variables (One-Hot Encoding drops the first column):', expanded=True):
		for col in cat_cols:	

				
			missing_values_method = st.radio("For the column '"+col+"':", ('Ordinal Encoding','One-Hot Encoding'))
			if missing_values_method == 'Ordinal Encoding':
				ordinal_encoder = OrdinalEncoder()

				try:
					X_train_cat2[col] = pd.DataFrame(ordinal_encoder.fit_transform(X_train_cat[col].values.reshape(-1,1))).reset_index(drop=True)
					try:
						X_valid_cat2[col] = pd.DataFrame(ordinal_encoder.transform(X_valid_cat[col].values.reshape(-1,1))).reset_index(drop=True)
					except:
						pass
				except:
					X_valid_cat2[col] = pd.DataFrame(ordinal_encoder.fit_transform(X_valid_cat[col].values.reshape(-1,1))).reset_index(drop=True)
				
			else:#one hot
				try:
					one_hot_encoder = OneHotEncoder(sparse=False,drop='first')
					new_col = pd.DataFrame(one_hot_encoder.fit_transform(X_train_cat[col].values.reshape(-1,1))).reset_index(drop=True)
					new_col.columns = one_hot_encoder.get_feature_names_out([col])
					
					X_train_cat2 = pd.concat([X_train_cat2, new_col], axis=1)
					X_train_cat2 = X_train_cat2.drop(col,axis=1)
					
					try:
						new_col = pd.DataFrame(one_hot_encoder.transform(X_valid_cat[col].values.reshape(-1,1))).reset_index(drop=True)
						new_col.columns = one_hot_encoder.get_feature_names_out([col])
						
						X_valid_cat2 = pd.concat([X_valid_cat2, new_col], axis=1)
						X_valid_cat2 = X_valid_cat2.drop(col,axis=1)
						
					except:
						pass
				except:
					new_col = pd.DataFrame(one_hot_encoder.fit_transform(X_valid_cat[col].values.reshape(-1,1))).reset_index(drop=True)
					new_col.columns = one_hot_encoder.get_feature_names([col])
					
					X_valid_cat2 = pd.concat([X_valid_cat2, new_col], axis=1)
					X_valid_cat2 = X_valid_cat2.drop(col,axis=1)
					


	st.title("The state of your data now:")
	st.dataframe(pd.concat([X_train_cat2,X_valid_cat2], axis=0))

	st.session_state['X_train'] = X_train_cat2
	st.session_state['X_valid'] = X_valid_cat2
	st.session_state['y_train'] = y_train
	st.session_state['y_valid'] = y_valid
def page3():
	
	st.title("Your data")




	X_train = st.session_state['X_train']
	X_valid = st.session_state['X_valid']
	y_train = st.session_state['y_train']
	y_valid = st.session_state['y_valid']


	st.dataframe(pd.concat([X_train,X_valid], axis=0))
	st.header("Data normalisation")
	norm_cols = dict()
	with st.expander(label='Which columns should be normalised?', expanded=True):
		for col in list(set(X_valid.columns) | set(X_train.columns)):
			norm_cols[col]=st.checkbox(label=f'{col}')
	
	scaler = preprocessing.StandardScaler()
	X_train_norm = X_train.copy()
	X_valid_norm = X_valid.copy()
	for col in list(set(X_valid.columns) | set(X_train.columns)):
		if(norm_cols[col]):
			try:
				X_train_norm[col]= scaler.fit_transform(X_train[col].values.reshape(-1,1))
				try:
					X_valid_norm[col]= scaler.transform(X_valid[col].values.reshape(-1,1))
				except Exception as e:
					print("===================================================================")
					print(e)
					pass 
			except:
				X_valid_norm[col]= scaler.fit_transform(X_valid[col].values.reshape(-1,1))

	st.header("Your training data:")
	st.dataframe(X_train_norm)	
	st.header("Your validation data:")
	st.dataframe(X_valid_norm)
	

	st.header("Choose your classification model")
	model_choice = st.selectbox("",("Logistic Regression","Nearest Neighbours","Decision Tree"),label_visibility="collapsed")
	
	models = {
	'Logistic Regression': LogisticRegression(),
	'Nearest Neighbours': KNeighborsClassifier(),
	'Decision Tree': DecisionTreeClassifier()
	}

	if(model_choice == "Logistic Regression"):
		st.write("A decision tree is a type of supervised learning algorithm that is used for classification and regression tasks. It creates a tree-like model of decisions and their possible consequences. Each internal node represents a test on a feature, each branch represents the outcome of the test, and each leaf node represents a class label or a numerical value. The algorithm uses a top-down, recursive approach to partition the data into smaller subsets based on the feature that provides the best split.")
	if(model_choice == 'Nearest Neighbours'):
		st.write("Logistic regression is a type of supervised learning algorithm used for classification tasks. It models the probability of the target variable (categorical) as a function of the predictor variables (continuous and/or categorical). The output is a probability value between 0 and 1, which is mapped to a binary class label based on a threshold. The algorithm estimates the parameters of the model using maximum likelihood estimation or gradient descent optimization.")
	if(model_choice == 'Decision Tree'):
		st.write("Nearest neighbors is a type of supervised learning algorithm used for classification and regression tasks. It is a non-parametric method that does not make any assumptions about the underlying distribution of the data. The algorithm works by finding the k nearest neighbors of a test instance in the training set based on a distance metric (e.g. Euclidean distance) and assigning the class label or regression value of the majority of the neighbors.")
	model = models[model_choice]
	model.fit(X_train_norm, y_train)
	preds = model.predict(X_valid_norm)
	preds_proba = model.predict_proba(X_valid_norm)[::,1]
	
	st.header("Precision, Recall, F-score and Support:")
	st.write("Precision, recall, F-score, and support are commonly used evaluation metrics in classification tasks. Precision measures the proportion of true positive labels among the predicted positive labels and reflects the accuracy of positive predictions. Recall measures the proportion of true positive labels that are correctly identified by the model and reflects the ability of the model to identify positive instances. The F-score combines precision and recall into a single metric and provides a more complete picture of the model's performance. Support represents the number of instances in each class in the dataset and is used to calculate the weighted average of the metrics. ")
	st.dataframe(pd.DataFrame(precision_recall_fscore_support(y_valid, preds)).set_index(pd.Series(['Precision', 'Recall', 'F-Score','Support'])))
	fpr, tpr, thresholds = roc_curve(y_valid, preds_proba)
	auc =roc_auc_score(y_valid, preds_proba)
	st.subheader("ROC Curve")
	st.write("The ROC curve and AUC are common evaluation metrics used to measure the performance of binary classification models. The ROC curve plots the true positive rate (TPR) against the false positive rate (FPR) for different classification thresholds, representing the trade-off between these rates. A perfect classifier would have a TPR of 1 and an FPR of 0, which would be located at the top left corner of the ROC curve. ")
	st.write(" The AUC is the area under the ROC curve, representing the overall performance of the classifier. An AUC of 1 indicates a perfect classifier, while an AUC of 0.5 indicates a random classifier.")
	fig, ax = plt.subplots()
	plt.plot(fpr, tpr,label="AUC="+str(auc))
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.legend(loc=4)
	st.pyplot(fig)
	##############
	# Initialize the RFE feature selector
	st.header("The most important features")
	rfe = RFE(estimator=model, n_features_to_select=10, step=1)
	rfe.fit(X_train_norm,y_train)
	st.write(list(X_train_norm.columns[rfe.ranking_ == 1]))
	st.write("To determine the most important features in the dataset, we use the feature selection method called Recursive Feature Elimination (RFE).")

	st.write("RFE works by recursively removing features from the dataset and building a model with the remaining features until the desired number of features is reached. The importance of each feature is ranked according to the order in which they are removed.")
	
	

st.set_page_config(page_title="ML Web Project", page_icon=":brain:", layout="wide")



page_names_to_funcs = {
	"Loading data": main,
	"Preprocessing data": page2,
	"Machine Learning": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()