import os
import glob
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library_tests.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(data_path):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = cls.import_data(data_path)

		assert df.shape[0] > 0
		assert df.shape[1] > 0

		logging.info("Testing import_data: SUCCESS")

	except FileNotFoundError as err:
		logging.error("Testing import_data: The file wasn't found")
		raise err
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err

def test_eda(data_path):
	'''
	test perform eda function
	'''

	# Arrange
	# Delete all previous eda files
	files = glob.glob("./images/eda/*")
	for file in files:
		os.remove(file)

	df = cls.import_data(data_path)

	# Act
	cls.perform_eda(df)

	# Assert
	try:
		assert os.path.isfile("./images/eda/Churn.png")
		assert os.path.isfile("./images/eda/Customer_Age.png")
		assert os.path.isfile("./images/eda/Marital_Status.png")
		assert os.path.isfile("./images/eda/Total_transactions.png")
		logging.info("Testing perform_eda: SUCCESS")
	except AssertionError as err:
		logging.error("Testing perform_eda: eda images not found")
		raise err



def test_encoder_helper(data_path, category_lst):
	'''
	test encoder helper
	'''
	df = cls.import_data(data_path)
	cls.perform_eda(df)
	new_df = cls.encoder_helper(df, category_lst)
	try:
		assert new_df is not None
		logging.info("Testing encoder_helper: SUCCESS")

	except AssertionError as err:
		logging.error("Testing encoder_helper: dataframe not returned")
		raise err

def test_perform_feature_engineering(data_path,category_lst):
	'''
	test perform_feature_engineering
	'''

	df = cls.import_data(data_path)
	cls.perform_eda(df)
	new_df = cls.encoder_helper(df, category_lst)

	X_train, X_test, X_test, y_test = cls.perform_feature_engineering(new_df)

	try:
		assert X_train is not None
		assert X_test is not None
		assert X_test is not None
		assert y_test is not None
		logging.info("Testing encoder_helper: SUCCESS")

	except AssertionError as err:
		logging.error("Testing perform_feature_engineering: no features returned")
		raise err

def test_train_models(data_path, category_lst):
	'''
	test train_models
	'''
	df = cls.import_data(data_path)
	cls.perform_eda(df)
	new_df = cls.encoder_helper(df, category_lst)

	X_train, X_test, X_test, y_test = cls.perform_feature_engineering(new_df)

	try:
		cls.train_models(X_train, X_test, X_test, y_test)

		assert os.path.isfile('./models/rfc_model.pkl')
		assert os.path.isfile('./models/logistic_model.pkl')
		logging.info("Testing train_models: SUCCESS")

	except AssertionError as err:
		logging.error("Testing train_models: model checkpoints not found")
		raise err

if __name__ == "__main__":
	print('Testing churn_library, please wait.')
	path = "./data/bank_data.csv"
	category_lst = ['Gender','Education_Level','Marital_Status', 'Income_Category','Card_Category']

	test_import(path)
	test_eda(path)
	test_encoder_helper(path, category_lst)
	test_perform_feature_engineering(path, category_lst)
	test_train_models(path, category_lst)
	print('Tests finished. For details check /logs/churn_library_tests.log')










