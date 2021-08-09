"""Functions to perform analysis on customer churn"""

import shap
import logging
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    data_frame['Churn'].hist()
    save_image('images/eda/Churn.png')

    data_frame['Customer_Age'].hist()
    save_image('images/eda/Customer_Age.png')

    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    save_image('images/eda/Marital_Status.png')

    sns.histplot(
        data_frame['Total_Trans_Ct'],
        kde=True,
        stat="density",
        linewidth=0)
    save_image('images/eda/Total_transactions.png')



def encoder_helper(data_frame, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        data_frame: pandas dataframe
        category_lst: list of columns that contain categorical features

    output:
        data_frame: pandas dataframe with new columns for
    '''
    for category in category_lst:
        lst = []
        groups = data_frame.groupby(category).mean()['Churn']

        for val in data_frame[category]:
            lst.append(groups.loc[val])
        new_column = category + '_Churn'

        data_frame[new_column] = lst

    return data_frame


def perform_feature_engineering(data_frame):
    '''
    input:
      data_frame: pandas dataframe

    output:
      X_train: X training data
      X_test: X testing data
      y_train: y training data
      y_test: y testing data
    '''
    y = data_frame['Churn']
    X = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = data_frame[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!

    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!

    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save figure to disk
    save_image(output_pth)

def analyze_classes_per_feature(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    fig = shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.margins()
    save_image('images/results/churn_vs_no_churn.png')

def save_image(path):
    """Save pyplot images to disk"""
    plt.savefig(path, bbox_inches='tight')
    plt.close(None)


def log_results(
        y_test,
        y_train,
        y_test_preds_rf,
        y_train_preds_rf,
        y_train_preds_lr,
        y_test_preds_lr):
    """
    Write classifiers results to log files.

    Input:
         y_test: y testing data
         y_train: y training data
         y_test_preds_rf: y predictions of the test dataset for random forest.
         y_train_preds_rf: y predictions of the train dataset for random forest.
         y_train_preds_lr: y predictions of the train dataset for logistic regression.
         y_test_preds_lr: y predictions of the test dataset for logistic regression.
    Output:
        None
    """
    logging.basicConfig(
        filename='./logs/results.log',
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')

    # scores
    logging.info('random forest results')
    logging.info('test results')
    logging.info(classification_report(y_test, y_test_preds_rf))
    logging.info('train results')
    logging.info(classification_report(y_train, y_train_preds_rf))

    logging.info('logistic regression results')
    logging.info('test results')
    logging.info(classification_report(y_test, y_test_preds_lr))
    logging.info('train results')
    logging.info(classification_report(y_train, y_train_preds_lr))


def save_roc_curves(lrc, random_forest_model, X_test, y_test):
    """Plot and save roc curve for models"""
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(random_forest_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    save_image('images/results/roc_curve.png')



def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='liblinear')

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Train random forest model
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # save best random forest model
    best_random_forest_model = cv_rfc.best_estimator_
    joblib.dump(best_random_forest_model, './models/rfc_model.pkl')

    y_train_preds_rf = best_random_forest_model.predict(X_train)
    y_test_preds_rf = best_random_forest_model.predict(X_test)

    # Train logistic regression model
    lrc.fit(X_train, y_train)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save logistic regression model
    joblib.dump(lrc, './models/logistic_model.pkl')

    log_results(
        y_test,
        y_train,
        y_test_preds_rf,
        y_train_preds_rf,
        y_train_preds_lr,
        y_test_preds_lr)

    feature_importance_plot(
        best_random_forest_model,
        X_train,
        'images/results/feature_importance_plot.png')

    save_roc_curves(lrc, best_random_forest_model, X_test, y_test)
    analyze_classes_per_feature(best_random_forest_model, X_test)
