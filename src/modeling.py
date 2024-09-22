import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from optbinning import OptimalPWBinning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, classification_report
from optbinning import OptimalBinning
from optbinning.scorecard import scorecard
from optbinning.binning import BinningProcess
from optbinning.scorecard import Scorecard
from imblearn.over_sampling import SMOTE
from optbinning.scorecard import plot_auc_roc, plot_cap, plot_ks
import os
import mlflow
from mlflow.models import signature, infer_signature



def load_data(data_path):
    """
    Helps loading data. tries csv and excel
    INPUT:
    ---------
    data_path
    ----------

    OUTPUT
    ---------
    - raw_data
    ---------
    """
    try:
        data = pd.read_csv(f'{data_path}')
        logging.info('SUCCESS LOADING DATA')
    except Exception as e:
        logging.error(f"error reading csv_file:{e}")

    try:
        data = pd.read_excel(f'{data_path}')
        logging.info('SUCCESS LOADING DATA')
    except Exception as e:
        logging.error(f"error reading excel_file:{e}")
    return data


def visualizations(data, wdir):
    """
    Data visualization
    INPUT:
    ---------
    - data
    - wdir (working directory)
    ----------

    OUTPUT
    ---------
    - charts (saves them into the outputs directory)
    ---------
    """
    for col in data.select_dtypes(include='number').columns:
        plt.figure()
        sns.kdeplot(data.loc[data['Default'] == 1,
                col], label='Default == 1')
        sns.kdeplot(data.loc[data['Default'] == 0,
                col], label='Default == 0')
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{wdir}/outputs/{col}_.png")

        return
    try: 
        for var in data.select_dtypes(include='number').columns:
            variable = var
            X = data[variable]
            y = data['Default'].values

            # Initialize the BinningProcess
            opt = OptimalBinning(name=variable, dtype="numerical",
                                class_weight="balanced", solver="mip", monotonic_trend="auto_asc_desc")
            opt.fit(X, y)
            binning_table = opt.binning_table
            binning_table.build()
            binning_table.plot(show_bin_labels=True)
            plt.savefig(f'{wdir}/outputs/{var}_bins.png')
    except Exception as e: 
        logging.error('Error ploting optimal binning::{e}')


def model_fitting(selection_criteria, list_variables, binning_fit_params, ml_model, X_train, y_train):
    """
    The function helps with model creation based on optimal binning
    INPUT:
    ---------
    - selection_criteria
    - list_variables
    - binning_fit_params
    - ml_model
    - training data (X_train and y_train)
    ----------

    OUTPUT
    ---------
    - fitted_model
    ---------
    """
    try:
        scaling_method = "min_max"
        scaling_method_data = {"min": 0, "max": 300}

        binning_process = BinningProcess(variable_names=list_variables,
            selection_criteria=selection_criteria,
            binning_fit_params=binning_fit_params)

        fitted_model = Scorecard(binning_process=binning_process,
                                estimator=ml_model, scaling_method=scaling_method,
                                scaling_method_params=scaling_method_data,
                                        intercept_based=False,
                                        reverse_scorecard=False)

        fitted_model.fit(X_train, y_train)

        return fitted_model
    except Exception as e:
        logging.error(f'Error in model fitting :: {e}')


def get_policy_table(score_data, config_file):
    """
    The function helps spliting the customers into different risk bands,
    compute possible loss, profits in each of the bands
    INPUT:
    ---------
    - score data
    - config file containing assumptions
    -
    ----------

    OUTPUT
    ---------
    - policy table
    ---------
    """
    try: 
        distrib_assumption = config_file['distribution_assumption']
        average_amount = config_file['average_amount']
        average_percentage_rate = config_file['average_percentage_rate']
        base_rate = config_file['base_rate']
        cost_of_bad_loan_multiplier = config_file['cost_of_bad_loan_multiplier']
        scores = score_data.groupby('customer_id').agg(
            score=pd.NamedAgg(column='overall_score', aggfunc='max'),
            default=pd.NamedAgg(column='Default', aggfunc='max')
            ).reset_index()

        scores = scores.sort_values(by='score', ascending=False)

        policy_tab = pd.DataFrame(list(distrib_assumption.items()), columns=[
            'risk_category', 'distribution'])

        policy_tab['cummulative_customers'] = policy_tab['distribution'].cumsum()

        policy_tab['cummulative_scores'] = round((
            policy_tab['cummulative_customers']/100)*scores[scores['score'] > 0].shape[0], 0)

        policy_tab['min_cut_off_score'] = None

        sorted_scores = sorted(scores[scores['score'] >
                                    0]['score'], reverse=True)
        policy_tab.loc[0, 'max_cut_off_score'] = scores['score'].max()

        for i in range(policy_tab.shape[0]):
            cut_off = int(policy_tab.loc[i, 'cummulative_scores'] - 1)
            min_score = sorted_scores[cut_off]
            policy_tab.loc[i, 'min_cut_off_score'] = min_score
            if i + 1 >= policy_tab.shape[0]:
                break
            policy_tab.loc[i + 1, 'max_cut_off_score'] = policy_tab.loc[i,
                                                                        'min_cut_off_score']

        policy_tab['max_cut_off_score'] = policy_tab['max_cut_off_score'].astype(
            int)
        policy_tab['max_cut_off_score'] = round(
            policy_tab['max_cut_off_score'].astype(int))
        policy_tab['min_cut_off_score'] = round(
            policy_tab['min_cut_off_score'].astype(int))

        def count_good_bads(row):
            scores_within_range = score_data[(score_data['overall_score'] > row['min_cut_off_score']) &
                                            (score_data['overall_score'] <= row['max_cut_off_score'])]
            count_goods = scores_within_range[scores_within_range['Default'] == 0]['Default'].count(
            )
            count_bads = scores_within_range[scores_within_range['Default'] == 1]['Default'].count(
            )
            return count_goods, count_bads

        policy_tab[['goods', 'bads']] = policy_tab.apply(
            count_good_bads, axis=1, result_type='expand')
        policy_tab['totals'] = policy_tab['bads'] + policy_tab['goods']
        policy_tab['bad_rate%'] = round(
            policy_tab['bads']*100/policy_tab['totals'], 1)
        for i in range(policy_tab.shape[0]):
            policy_tab.loc[i, 'bad_rate_at_cut_off%'] = round(
                sum(policy_tab.loc[0:i, 'bads'])*100/sum(policy_tab.loc[0:i, 'totals']), 1)
            policy_tab.loc[i, 'perc_contrast'] = round(
                policy_tab.loc[i, 'totals']*100/sum(policy_tab['totals']), 1
            )
            policy_tab.loc[i, 'cumu_perc_contrast'] = round(
                sum(policy_tab.loc[0:i, 'totals'])*100/sum(policy_tab['totals']), 1
            )
            policy_tab.loc[i, 'cumu_perc_goods'] = round(
                sum(policy_tab.loc[0:i, 'goods'])*100/sum(policy_tab['goods']), 1
            )
            policy_tab.loc[i, 'cumu_perc_bads'] = round(
                sum(policy_tab.loc[0:i, 'bads'])*100/sum(policy_tab['bads']), 1
            )

        policy_tab['average_amount'] = average_amount
        policy_tab['average_margin'] = round(average_percentage_rate-base_rate, 1)
        policy_tab['Interest Income'] = round(
            policy_tab['goods']*policy_tab['average_amount']*policy_tab['average_margin']/100, 0)

        policy_tab['Ave. Total Charge Off'] = round(
            policy_tab['average_amount']*cost_of_bad_loan_multiplier)
        policy_tab['Total Loss'] = round(
            policy_tab['bads']*policy_tab['average_amount'], 0)
        policy_tab['Gross Margin'] = policy_tab['Interest Income'] - \
            policy_tab['Total Loss']

        return policy_tab

    except Exception as e:
        logging.error(f'Error generating policy table:: {e}')


# Function to create an experiment in MLFlow and log parameters, metrics and artifacts files like images etc.

def create_experiment(wdir, model_data, model_name, experiment_name, run_name, run_metrics, model,
                      run_params=None,y=None, X=None):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        dataset_source_url = "https://www.kaggle.com/datasets/neelesh0602/bankcsv"
        

        dataset: PandasDataset = mlflow.data.from_pandas(
            model_data, source=dataset_source_url)

        mlflow.log_input(dataset, context="training")
        signature = infer_signature(
            X, model.predict(X))
        #log model
        mlflow.sklearn.log_model(
            model, model_name, signature=signature)

        ######
        score = model.score(X)
        mask = y == 0
        plt.hist(score[mask], label="non-event", color="b", alpha=0.35)
        plt.hist(score[~mask], label="event", color="r", alpha=0.35)
        plt.xlabel("score")
        plt.legend()
        plt.savefig(f"{wdir}/outputs/DEFAULT_DISTRIB.png")
        mlflow.log_artifact(f"{wdir}/outputs/DEFAULT_DISTRIB.png", artifact_path="outputs")
        preds = model.predict(X)

        cm = confusion_matrix(y, preds)
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, )

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
            'Non_Default', 'Default'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=15, pad=20)
        plt.xlabel('Prediction', fontsize=11)
        plt.ylabel('Actual', fontsize=11)
        # Customizations
        plt.gca().xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()
        plt.gca().figure.subplots_adjust(bottom=0.2)
        plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
        plt.savefig(f'{wdir}/outputs/confusion_matrix.png')
        # mlflow.log_artifact(f"{wdir}/outputs/confusion_matrix.png", artifact_path="outputs")

        f1_scor = f1_score(y, preds)
        mlflow.log_metric("f1_score", f1_scor)
        
        recall_scor = recall_score(y, preds)
        mlflow.log_metric("recall_score", recall_scor)
        precision_scor = precision_score(y, preds)
        mlflow.log_metric("precision_score", precision_scor)


            #params
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])

        # for metric in run_metrics:
        #     mlflow.log_metric(metric, run_metrics[metric])

            # Log model
        signature = infer_signature(
            X, model.predict(X))

        mlflow.sklearn.log_model(model, "model", signature=signature)

        mlflow.sklearn.log_model(model, "model")

        mlflow.log_artifact(f'{wdir}/outputs/confusion_matrix.png',
                            artifact_path="performance")
        mlflow.log_artifact(f"{wdir}/outputs/auc_roc.png",
                            artifact_path="performance")

        mlflow.log_artifact(f'{wdir}/outputs/confusion_matrix.png',
                            artifact_path="performance")

        mlflow.set_tag("tag1", "logistic_regression_model")
    print('Run - %s is logged to Experiment - %s' %
          (run_name, experiment_name))


# creating the MLflow

def main():
    wdir = os.getcwd()
    data_path = f'{wdir}/outputs/data_for_modeling.csv'
    model_data = load_data(data_path)
    viz = visualizations(model_data,wdir)


    selection_criteria = {
        "iv": {"min": 0.02, "max": 0.5},
        "quality_score": {"min": 0.0001}
        }


    list_variables = ['Class of Business', 'Transaction Type', 'Sub-Class of Business',
                        'Amount', 'Income', 'Number of Claims', 'Account Balance',
                         'income_balance_ratio', 'inactive_days',
                        #  'Amount_converted'
                        ]


    binning_fit_params = {
        "Class of Business": {"dtype": "categorical", "solver": "mip", "class_weight": "balanced"},
        "Transaction Type": {"dtype": "categorical", "solver": "mip", "class_weight": "balanced"},
        "Sub-Class of Business": {"dtype": "categorical", "solver": "mip", "class_weight": "balanced"},
        "Amount": {"dtype": "numerical", "solver": "mip", "class_weight": "balanced"},
        "Income": {"dtype": "numerical", "solver": "mip", "class_weight": "balanced"},
        "Number of Claims": {"dtype": "numerical", "solver": "mip", "class_weight": "balanced"},
        "Account Balance": {"dtype": "numerical", "solver": "mip", "class_weight": "balanced"},
        "income_balance_ratio": {"dtype": "numerical", "solver": "mip", "class_weight": "balanced"},
        "inactive_days": {"dtype": "numerical", "solver": "mip", "class_weight": "balanced"},
        # "Amount_converted": {"dtype": "numerical", "solver": "mip", "class_weight": "balanced"}
    }
    X = model_data[list_variables]
    y = model_data['Default'].values

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=321)
    from imblearn.over_sampling import RandomOverSampler


    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_train, y_train = oversampler.fit_resample(X, y)

    ml_model = LogisticRegression(C=3, max_iter=1000, random_state=42)

    fitted_model = model_fitting(selection_criteria,
     list_variables, binning_fit_params, ml_model, X_train, y_train)

    plt.figure()
    pred_proba = fitted_model.predict_proba(X_test)[:, 1]
    plot_auc_roc(y_test, pred_proba)
    plt.savefig(f'{wdir}/outputs/auc_roc.png')


    data = model_data.copy()
    data['overall_score'] = fitted_model.score(data)
    data['Default'] = data['Default']
    policy_table_assumptions_config_file = {
        "distribution_assumption": {
            "low_risk": 5,
            "medium_low_risk": 15,
            "medium_risk": 30,
            "medium_high_risk": 25,
            "high_risk": 15,
            "very_high_risk": 10
        },
        "average_amount": 2272,
        "average_percentage_rate": 13,
        "base_rate": 0,
        "cost_of_bad_loan_multiplier": 1
    }
    policy_tab = get_policy_table(
        score_data=data, config_file=policy_table_assumptions_config_file)
    policy_tab.to_csv(f"{wdir}/outputs/policy_table.csv", index=False)

    # print(
    #     f"The f1_score::{f1_scor} the recall_score::{recall_scor} the precision_score::{precision_scor}")

    experiment_name = 'credit_scorer'
    run_name = 'first_gen_model'
    model_name = 'ML_scorecard'
    run_metrics=None
    create_experiment(experiment_name=experiment_name, model_name=model_name, run_name=run_name, run_metrics=run_metrics,
                      model=fitted_model, model_data=model_data, X=X_test, y=y_test, wdir=wdir)


if __name__ == "__main__":
    main()
