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

	input_header = st.number_input("Enter the row number of the header, or leave empty for automatic detection:",step=1, min_value=0)

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

	target_col=st.radio("Choose the Target column:",df.columns, help="This is the value that will be predicted by the model you build")

	y = df[target_col]
	ordinal_encoder = OrdinalEncoder()
	y = pd.DataFrame(ordinal_encoder.fit_transform(y.values.reshape(-1,1))).reset_index(drop=True)
	X = df.drop([target_col], axis=1)

	cat_cols = dict()
	with st.expander(label='Drop some of the columns?', expanded=True):
		for col in X.columns:
			cat_cols[col]=st.checkbox(label=f'{col}')
	
	X2=X.copy()
	del_cols = []
	for col in X.columns:
		if(cat_cols[col]):
			del_cols.append(col)

	X2 = X.drop(del_cols,axis=1)

	train_size = st.slider('What is the percentage of training data in your dataset?', 1, 99, 80)
	train_size = train_size /100

	X_train, X_valid, y_train, y_valid = train_test_split(X2, y, train_size=train_size, test_size=1-train_size,
	                                                            random_state=0)

	##############

	special_char = st.text_input("Enter any special strings that are considered a Null Value (use commas to separate)")
	special_char= special_char.split(",")
	for char in special_char:
		X_train = X_train.replace(char, np.nan)
		X_valid = X_valid.replace(char, np.nan)

	st.title("Dealing with missing values")

	with st.expander(label='Number of Missing Values:', expanded=True):
		st.write(pd.concat([X_train, X_valid], axis=0).isna().sum())

	#radio_missing_values = st.radio("How to deal with the missing values?", ('Drop the rows containing Null values', 'Choose the method for each column'))
	del_cols = []
	# if radio_missing_values == 'Drop the rows containing Null values':
	# 	df2 = df.dropna()
	#else: ##method for each column
	missing_values_cols = X2.columns[X2.isna().any()].tolist()
	st.title("Method to deal with missing values")
	st.write("Note: Imputation picks the average if numeric values or the most commun if categorical")
	X_train=X_train.reset_index(drop=True)
	X_valid=X_valid.reset_index(drop=True)
	X_train_no_miss= X_train.copy()
	X_valid_no_miss = X_valid.copy()
	
	for col in missing_values_cols:	
		
		missing_values_method = st.radio("For the column '"+col+"':", ('Drop the column', 
			'mean Imputation','mean Imputation + create a new column to indicate missing value'
			,'most frequent Imputation','most frequent Imputation + create a new column to indicate missing value'))
		if missing_values_method == 'Drop the column':
			del_cols.append(col)
		
		elif missing_values_method == 'most frequent Imputation':
			try:
				my_imputer = SimpleImputer(strategy='most_frequent')
				X_train_no_miss[col] = pd.DataFrame(my_imputer.fit_transform(X_train[col].values.reshape(-1,1))).reset_index(drop=True)
				X_valid_no_miss[col] = pd.DataFrame(my_imputer.transform(X_valid[col].values.reshape(-1,1))).reset_index(drop=True)
				#df2[col] = pd.DataFrame(my_imputer.fit_transform(df2[col].values.reshape(-1,1))).reset_index(drop=True)
			except Exception as e:
				st.error("Could not perform Imputation.")
				st.exception(e)
				
		elif missing_values_method == 'mean Imputation':				
			try:
				my_imputer = SimpleImputer(strategy='mean')
				X_train_no_miss[col] = pd.DataFrame(my_imputer.fit_transform(X_train[col].values.reshape(-1,1))).reset_index(drop=True)
				X_valid_no_miss[col] = pd.DataFrame(my_imputer.transform(X_valid[col].values.reshape(-1,1))).reset_index(drop=True)
				#df2[col] = pd.DataFrame(my_imputer.fit_transform(df2[col].values.reshape(-1,1))).reset_index(drop=True)
			except Exception as e:
				st.error("Could not perform Imputation.")
				st.exception(e)
		elif missing_values_method=='most frequent Imputation + create a new column to indicate missing value': 
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
	cat_cols = dict()
	X_train_cat = X_train_no_miss.copy().reset_index(drop=True)
	X_valid_cat = X_valid_no_miss.copy().reset_index(drop=True)
	#df3 = df3.reset_index(drop=True)
	with st.expander(label='Choose the categorical variables:', expanded=True):
		# l = len(list(set(X_valid_cat.columns) | set(X_train_cat.columns)))
		# l = range(l)
		for idx,col in enumerate(list(set(X_valid_cat.columns) | set(X_train_cat.columns))):
			cat_cols[col]=st.checkbox(label=f'{col}',key=idx)

	X_valid_cat2=X_valid_cat.reset_index(drop=True)
	X_train_cat2=X_train_cat.reset_index(drop=True)
	# cat_trans_method = st.radio("Choose the method to deal with categorical variables:",('Ordinal Encoding','One-Hot Encoding'))
	with st.expander(label='Dealing with the categorical variables (ps: One-Hot Encoding drops the first column):', expanded=True):
		for col in cat_cols:	

			if(cat_cols[col]):
				
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
						#X_train_cat2 = X_train_cat.drop(col,axis=1)
						X_train_cat2 = pd.concat([X_train_cat, new_col], axis=1)
						X_train_cat2 = X_train_cat2.drop(col,axis=1)
						try:
							new_col = pd.DataFrame(one_hot_encoder.transform(X_valid_cat[col].values.reshape(-1,1))).reset_index(drop=True)
							new_col.columns = one_hot_encoder.get_feature_names_out([col])
							# X_valid_cat2 = X_valid_cat.drop(col,axis=1)
							X_valid_cat2 = pd.concat([X_valid_cat, new_col], axis=1)
							X_valid_cat2 = X_valid_cat2.drop(col,axis=1)
						except:
							pass 
					except:
						new_col = pd.DataFrame(one_hot_encoder.fit_transform(X_valid_cat[col].values.reshape(-1,1))).reset_index(drop=True)
						new_col.columns = one_hot_encoder.get_feature_names([col])
						#X_valid_cat2 = X_valid_cat.drop(col,axis=1)
						X_valid_cat2 = pd.concat([X_valid_cat, new_col], axis=1)
						X_valid_cat2 = X_valid_cat2.drop(col,axis=1)
					#########################
	# if cat_trans_method == "Ordinal Encoding":
		
	# 	ordinal_encoder = OrdinalEncoder()
	# 	for col in df2.columns:
	# 		if(cat_cols[col]):
	# 			df3[col] = pd.DataFrame(ordinal_encoder.fit_transform(df2[col].values.reshape(-1,1))).reset_index(drop=True)
				
	# else:
	# 	one_hot_encoder = OneHotEncoder(sparse=False)
	# 	for col in df2.columns:
	# 		if(cat_cols[col]):
				
	# 			new_col = pd.DataFrame(one_hot_encoder.fit_transform(df2[col].values.reshape(-1,1))).reset_index(drop=True)
	# 			new_col.columns = one_hot_encoder.get_feature_names([col])
	# 			df3 = df3.drop(col,axis=1)
	# 			df3 = pd.concat([df3, new_col], axis=1)

	st.title("The state of your data now:")
	st.dataframe(pd.concat([X_valid_cat2, X_train_cat2], axis=0))

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

	# cat_cols = dict()
	# with st.expander(label='Final touches, Choose any columns you want to drop:', expanded=True):
	# 	for col in df.columns:
	# 		cat_cols[col]=st.checkbox(label=f'{col}')
	
	# df2=df.copy()
	# del_cols = []
	# for col in df.columns:
	# 	if(cat_cols[col]):
	# 		del_cols.append(col)

	# df2 = df.drop(del_cols,axis=1)

	st.write("Training Data:")
	st.dataframe(X_train)	
	st.write("Validation Data:")
	st.dataframe(X_valid)
	

	norm_cols = dict()
	with st.expander(label='cols to normalise:', expanded=True):
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

	st.write("Your training data now:")
	st.dataframe(X_train_norm)	
	st.write("Your validation data now:")
	st.dataframe(X_valid_norm)
	

	# norm_scale=st.radio("Normalisation or Scaling?:",("Normalisation","Scaling"))
	# if norm_scale == "Scaling":
	# 	X = minmax_scaling(X)
	# else:
	# 	X = stats.boxcox(X)

	# st.dataframe(X)
		
	# train_size = st.slider('What is the percentage of training data in your dataset?', 1, 99, 80)
	# train_size = train_size /100

	# X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=train_size, test_size=1-train_size,
 #                                                                random_state=0)

	
	model_choice = st.radio("Choose your classification model:",("Logistic Regression","Nearest Neighbours","Decision Tree"))
	
	models = {
	'Logistic Regression': LogisticRegression(),
	'Nearest Neighbours': KNeighborsClassifier(),
	'Decision Tree': DecisionTreeClassifier()
	}

	model = models[model_choice]
	model.fit(X_train_norm, y_train)
	preds = model.predict(X_valid_norm)
	preds_proba = model.predict_proba(X_valid_norm)[::,1]
	
	st.write("Precision, Recall, F-score and Support:")
	st.dataframe(pd.DataFrame(precision_recall_fscore_support(y_valid, preds)).set_index(pd.Series(['Precision', 'Recall', 'F-Score','Support'])))
	fpr, tpr, thresholds = roc_curve(y_valid, preds_proba)
	auc =roc_auc_score(y_valid, preds_proba)
	st.subheader("ROC Curve")
	fig, ax = plt.subplots()
	plt.plot(fpr, tpr,label="AUC="+str(auc))
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.legend(loc=4)
	st.pyplot(fig)
	

st.set_page_config(page_title="File Reader", page_icon=":guardsman:", layout="wide")



page_names_to_funcs = {
	"Loading data": main,
	"Preprocessing data": page2,
	"Machine Learning": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()