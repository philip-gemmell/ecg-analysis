# Conduct random forest analysis - need data from analysis_commands.txt to be able to run

# from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.tree import export_graphviz
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import pydot
from tabulate import tabulate

RSEED = 42


def plot_confusion_matrix(cm_data, class_names, normalize=False, title='Confusion matrix', cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm_data = cm_data.astype('float') / cm_data.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm_data)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm_data, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, size=14)
    plt.yticks(tick_marks, class_names, size=14)

    fmt = '.2f' if normalize else 'd'
    thresh_cm = cm_data.max() / 2.

    # Labeling the plot
    for j, k in itertools.product(range(cm_data.shape[0]), range(cm_data.shape[1])):
        plt.text(k, j, format(cm_data[j, k], fmt), fontsize=20, horizontalalignment="center", color="white" if
        cm_data[j, k] > thresh_cm else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)


def evaluate_model(predictions_, probs, train_predictions_, train_probs, test_labels_, train_labels_, title):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""

    baseline = dict()
    baseline['recall'] = recall_score(test_labels_, [1 for _ in range(len(test_labels_))])
    baseline['precision'] = precision_score(test_labels_, [1 for _ in range(len(test_labels_))])
    baseline['roc'] = 0.5

    results = dict()
    results['recall'] = recall_score(test_labels_, predictions_)
    results['precision'] = precision_score(test_labels_, predictions_)
    results['roc'] = roc_auc_score(test_labels_, probs)

    train_results = dict()
    train_results['recall'] = recall_score(train_labels_, train_predictions_)
    train_results['precision'] = precision_score(train_labels_, train_predictions_)
    train_results['roc'] = roc_auc_score(train_labels_, train_probs)

    for metric in ['recall', 'precision', 'roc']:
        print('{} Baseline: {} Test: {} Train: {}'.format(metric.capitalize(), round(baseline[metric], 2),
                                                          round(results[metric], 2), round(train_results[metric], 2)))

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels_, [1 for _ in range(len(test_labels_))])
    model_fpr, model_tpr, _ = roc_curve(test_labels_, probs)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ' ROC Curves')


# # Construct series of simulated VCGs by adding noise
# n_noise = 100
# temp=list()
# for vcg in vcg_lv_phi_300:
# 	for i in range(n_noise):
# 		temp.append(vcg+np.random.normal(0, 0.05, vcg.shape)
# vcg_lv_phi_300 = temp[:]
# temp=list()
# for vcg in vcg_lv_phi_600:
# 	for i in range(n_noise):
# 		temp.append(vcg+np.random.normal(0, 0.05, vcg.shape)
# vcg_lv_phi_600 = temp[:]
# temp=list()
# for vcg in vcg_lv_rho_300:
# 	for i in range(n_noise):
# 		temp.append(vcg+np.random.normal(0, 0.05, vcg.shape)
# vcg_lv_rho_300 = temp[:]
# temp=list()
# for vcg in vcg_lv_rho_600:
# 	for i in range(n_noise):
# 		temp.append(vcg+np.random.normal(0, 0.05, vcg.shape)
# vcg_lv_rho_600 = temp[:]
# temp=list()
# for vcg in vcg_lv_z_300:
# 	for i in range(n_noise):
# 		temp.append(vcg+np.random.normal(0, 0.05, vcg.shape)
# vcg_lv_z_300 = temp[:]
# temp=list()
# for vcg in vcg_lv_z_600:
# 	for i in range(n_noise):
# 		temp.append(vcg+np.random.normal(0, 0.05, vcg.shape)
# vcg_lv_z_600 = temp[:]
# temp=list()
# for vcg in vcg_lv_size_300:
# 	for i in range(n_noise):
# 		temp.append(vcg+np.random.normal(0, 0.05, vcg.shape)
# vcg_lv_size_600 = temp[:]
# temp=list()
# for vcg in vcg_lv_z_600:
# 	for i in range(n_noise):
# 		temp.append(vcg+np.random.normal(0, 0.05, vcg.shape)
# vcg_lv_size_600 = temp[:]
# temp=list()
# for vcg in vcg_lv_other_300:
# 	for i in range(n_noise):
# 		temp.append(vcg+np.random.normal(0, 0.05, vcg.shape)
# vcg_lv_other_300 = temp[:]
# temp=list()
# for vcg in vcg_lv_other_600:
# 	for i in range(n_noise):
# 		temp.append(vcg+np.random.normal(0, 0.05, vcg.shape)
# vcg_lv_other_600 = temp[:]
#
#
# vcg_septum_phi_300 = vcg_analysis.convert_ecg_to_vcg(ecg_septum_phi_300)
# vcg_septum_phi_600 = vcg_analysis.convert_ecg_to_vcg(ecg_septum_phi_600)
# vcg_septum_rho_300 = vcg_analysis.convert_ecg_to_vcg(ecg_septum_rho_300)
# vcg_septum_rho_600 = vcg_analysis.convert_ecg_to_vcg(ecg_septum_rho_600)
# vcg_septum_z_300 = vcg_analysis.convert_ecg_to_vcg(ecg_septum_z_300)
# vcg_septum_z_600 = vcg_analysis.convert_ecg_to_vcg(ecg_septum_z_600)
# vcg_septum_size_300 = vcg_analysis.convert_ecg_to_vcg(ecg_septum_size_300)
# vcg_septum_size_600 = vcg_analysis.convert_ecg_to_vcg(ecg_septum_size_600)
# vcg_septum_other_300 = vcg_analysis.convert_ecg_to_vcg(ecg_septum_other_300)
# vcg_septum_other_600 = vcg_analysis.convert_ecg_to_vcg(ecg_septum_other_600)

# Collate data for input parameters (the values we want to be able to predict!)
parameters = ['lv', 'septum', 'volume', 'area', 'phi', 'rho', 'z']
params_classify = ['lv', 'septum']
df_params = pd.DataFrame(columns=parameters)
df_params[parameters[0]] = [1 for _ in range(len(area_lv))] + [0 for _ in range(len(area_septum))]
df_params[parameters[1]] = [0 for _ in range(len(area_lv))] + [1 for _ in range(len(area_septum))]
df_params[parameters[2]] = volume_complete
# Need to adjust lv/septum variables for when there is no scar i.e. control case
for i in range(len(volume_complete)):
    if volume_complete[i] == 0.0:
        df_params.loc[i, parameters[0]] = 0
        df_params.loc[i, parameters[1]] = 0
df_params[parameters[3]] = area_complete_norm
df_params[parameters[4]] = rangePhi_complete
df_params[parameters[5]] = rangeRho_complete
df_params[parameters[6]] = rangeZ_complete

# Collate the output data metrics (don't use QRS Area (3D) or dTQRS075)
metrics = ['qrs_duration', 'qrs_area_pythag', 'waa', 'wae', 'weightMag', 'maxMag', 'maxMagTime',
           'dTuwd', 'dTmax', 'dTqrs000', 'dTqrs050', 'dTqrs100'
           # , 'dTqrs075'
           ]
metrics_important = ['qrs_duration', 'qrs_area_pythag', 'maxMag', 'dTuwd', 'dTmax', 'dTqrs100']
df_metrics = pd.DataFrame(columns=metrics)
df_metrics[metrics[0]] = delta_qrs_complete
df_metrics[metrics[1]] = delta_qrs_area_frac_pythag_complete
df_metrics[metrics[2]] = delta_waaAngle_complete
df_metrics[metrics[3]] = delta_waeAngle_complete
df_metrics[metrics[4]] = delta_weightMag_frac_complete
df_metrics[metrics[5]] = delta_maxMag_frac_complete
df_metrics[metrics[6]] = delta_maxMagTime_complete
df_metrics[metrics[7]] = dtAngle_complete
df_metrics[metrics[8]] = dtMaxAngle_complete
df_metrics[metrics[9]] = dtQRS000Angle_complete
df_metrics[metrics[10]] = dtQRS050Angle_complete
df_metrics[metrics[11]] = dtQRS100Angle_complete
# df_metrics[metrics[12]] = dtQRS075Angle_complete

# Drop repeats of the control case (no scar)
noscar = df_params.index[df_params['volume'] == 0].to_list()[1:]
noscar_nocontrol = df_params.index[df_params['volume'] == 0].to_list()
df_params_nocontrol = df_params.drop(noscar_nocontrol, inplace=False)
df_metrics_nocontrol = df_metrics.drop(noscar_nocontrol, inplace=False)
np_metrics_nocontrol = np.array(df_metrics_nocontrol)
df_params.drop(noscar, inplace=True)
df_metrics.drop(noscar, inplace=True)
np_metrics = np.array(df_metrics)

# Construct random forests, and generate predictions for test data
train_features = dict.fromkeys(parameters)
test_features = dict.fromkeys(parameters)
train_labels = dict.fromkeys(parameters)
test_labels = dict.fromkeys(parameters)
rf = dict.fromkeys(parameters)
predictions = dict.fromkeys(parameters)
probabilities = dict.fromkeys(params_classify)
train_predictions = dict.fromkeys(parameters)
train_probabilities = dict.fromkeys(params_classify)
oob_accuracy = dict.fromkeys(parameters)
for key in parameters:
    train_features[key], test_features[key], train_labels[key], test_labels[key] = \
        train_test_split(np_metrics, np.array(df_params[key]), test_size=0.3, random_state=42)
    if key in params_classify:
        rf[key] = RandomForestClassifier(n_estimators=1000, random_state=RSEED, n_jobs=-4, oob_score=True)
    else:
        rf[key] = RandomForestRegressor(n_estimators=1000, random_state=RSEED, n_jobs=-4, oob_score=True)
    rf[key].fit(train_features[key], train_labels[key])
    oob_accuracy[key] = rf[key].oob_score_

# Calculate predictions of model, and probabilities of these predictions (along with check on the predictions and
# probabilities of the training data, to check for overfitting
for key in parameters:
    predictions[key] = rf[key].predict(test_features[key])
    train_predictions[key] = rf[key].predict(train_features[key])
    if key in params_classify:
        probabilities[key] = rf[key].predict_proba(test_features[key])[:, 1]
        train_probabilities[key] = rf[key].predict_proba(train_features[key])[:, 1]

# Check out-of-bag errors against number of trees in the forest to assess required size
sizeCheck_rf = dict.fromkeys(parameters)
sizeCheck_oob = dict.fromkeys(parameters)
min_estimators = 5
max_estimators = 60
for key in parameters:
    if key in params_classify:
        sizeCheck_rf[key] = RandomForestClassifier(random_state=RSEED, n_jobs=-2, oob_score=True)
    else:
        sizeCheck_rf[key] = RandomForestRegressor(random_state=RSEED, n_jobs=-2, oob_score=True)
    sizeCheck_oob[key] = list()
    for i in range(min_estimators, max_estimators + 1):
        sizeCheck_rf[key].set_params(n_estimators=i)
        sizeCheck_rf[key].fit(np_metrics, np.array(df_params[key]))
        sizeCheck_oob[key].append((i, 1-sizeCheck_rf[key].oob_score_))
for label, clf_err in sizeCheck_oob.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend()
plt.show()

# Conduct cross-validation using k-fold cross-validation (use entire data-set, not just training)
# https://scikit-learn.org/stable/modules/cross_validation.html
cv5_scores20 = dict.fromkeys(parameters)
cvn_scores20 = dict.fromkeys(parameters)
rf_full20 = dict.fromkeys(parameters)
cv5_scores1000 = dict.fromkeys(parameters)
cvn_scores1000 = dict.fromkeys(parameters)
rf_full1000 = dict.fromkeys(parameters)
cv5_scores20_nocontrol = dict.fromkeys(parameters)
cvn_scores20_nocontrol = dict.fromkeys(parameters)
rf_full20_nocontrol = dict.fromkeys(parameters)
cv5_scores1000_nocontrol = dict.fromkeys(parameters)
cvn_scores1000_nocontrol = dict.fromkeys(parameters)
rf_full1000_nocontrol = dict.fromkeys(parameters)
for key in parameters:
    if key in params_classify:
        rf_full20[key] = RandomForestClassifier(random_state=RSEED, n_jobs=-2, oob_score=True)
        rf_full1000[key] = RandomForestClassifier(random_state=RSEED, n_jobs=-2, oob_score=True)
        rf_full20_nocontrol[key] = RandomForestClassifier(random_state=RSEED, n_jobs=-2, oob_score=True)
        rf_full1000_nocontrol[key] = RandomForestClassifier(random_state=RSEED, n_jobs=-2, oob_score=True)
    else:
        rf_full20[key] = RandomForestRegressor(random_state=RSEED, n_jobs=-2, oob_score=True)
        rf_full1000[key] = RandomForestRegressor(random_state=RSEED, n_jobs=-2, oob_score=True)
        rf_full20_nocontrol[key] = RandomForestRegressor(random_state=RSEED, n_jobs=-2, oob_score=True)
        rf_full1000_nocontrol[key] = RandomForestRegressor(random_state=RSEED, n_jobs=-2, oob_score=True)
    rf_full20[key].set_params(n_estimators=20)
    rf_full20[key].fit(np_metrics, np.array(df_params[key]))
    rf_full1000[key].set_params(n_estimators=1000)
    rf_full1000[key].fit(np_metrics, np.array(df_params[key]))
    rf_full20_nocontrol[key].set_params(n_estimators=20)
    rf_full20_nocontrol[key].fit(np_metrics_nocontrol, np.array(df_params_nocontrol[key]))
    rf_full1000_nocontrol[key].set_params(n_estimators=1000)
    rf_full1000_nocontrol[key].fit(np_metrics_nocontrol, np.array(df_params_nocontrol[key]))
    cv5_scores20[key] = cross_val_score(rf_full20[key], np_metrics, np.array(df_params[key]), cv=5, n_jobs=-2)
    cvn_scores20[key] = cross_val_score(rf_full20[key], np_metrics, np.array(df_params[key]), cv=len(parameters),
                                        n_jobs=-2)
    cv5_scores1000[key] = cross_val_score(rf_full1000[key], np_metrics, np.array(df_params[key]), cv=5, n_jobs=-2)
    cvn_scores1000[key] = cross_val_score(rf_full1000[key], np_metrics, np.array(df_params[key]), cv=len(parameters),
                                          n_jobs=-2)
    cv5_scores20_nocontrol[key] = cross_val_score(rf_full20_nocontrol[key], np_metrics_nocontrol,
                                                  np.array(df_params_nocontrol[key]), cv=5, n_jobs=-2)
    cvn_scores20_nocontrol[key] = cross_val_score(rf_full20_nocontrol[key], np_metrics_nocontrol,
                                                  np.array(df_params_nocontrol[key]), cv=len(parameters), n_jobs=-2)
    cv5_scores1000_nocontrol[key] = cross_val_score(rf_full1000_nocontrol[key], np_metrics_nocontrol,
                                                    np.array(df_params_nocontrol[key]), cv=5, n_jobs=-2)
    cvn_scores1000_nocontrol[key] = cross_val_score(rf_full1000_nocontrol[key], np_metrics_nocontrol,
                                                    np.array(df_params_nocontrol[key]), cv=len(parameters), n_jobs=-2)

# Extract details for the mean depth of the trees, etc.
n_nodes = dict.fromkeys(parameters)
depth = dict.fromkeys(parameters)
for key in rf:
    n_nodes[key] = list()
    depth[key] = list()
    for tree in rf[key].estimators_:
        n_nodes[key].append(tree.tree_.node_count)
        depth[key].append(tree.tree_.max_depth)

# Calculate errors, and from that the accuracy - use several different methods, including cross-validation
# https://datascience.stackexchange.com/questions/13151/randomforestclassifier-oob-scoring-method
errors = dict.fromkeys(parameters)
accuracy = dict.fromkeys(parameters)
for key in parameters:
    # Personal calculation
    errors[key] = abs(predictions[key] - test_labels[key])
    if key in params_classify:
        accuracy[key] = 100 - sum(errors[key])/len(errors[key]) * 100
    else:
        # Calculate mean absolute percentage error (MAPE)
        accuracy[key] = 100-np.mean(100*(errors[key]/test_labels[key]))

# Check ROC AUC scores (including for training predictions), and plot curves
train_roc_auc = dict.fromkeys(parameters)
test_roc_auc = dict.fromkeys(parameters)
fpr = dict.fromkeys(params_classify)  # False positive rate
tpr = dict.fromkeys(params_classify)  # True positive rate
thresh = dict.fromkeys(params_classify)  # Thresholds used to compute fpr and tpr
fig = plt.figure()
ax = fig.add_subplot(111)
linestyle = {'lv': '-', 'septum': '--'}
cm = dict.fromkeys(params_classify)
classes = {'lv': ['Septum', 'LV'], 'septum': ['LV', 'Septum']}
for key in params_classify:
    # evaluate_model(predictions[key], probabilities[key], train_predictions[key], train_probabilities[key],
    #                test_labels[key], train_labels[key], key)
    # cm[key] = confusion_matrix(test_labels[key], predictions[key])
    # plot_confusion_matrix(cm[key], class_names=classes[key], title=key + ' Scar Classification')
    train_roc_auc[key] = roc_auc_score(train_labels[key], train_probabilities[key])
    test_roc_auc[key] = roc_auc_score(test_labels[key], probabilities[key])
    fpr[key], tpr[key], thresh[key] = roc_curve(test_labels[key], probabilities[key])
    ax.plot(fpr[key], tpr[key], label=key, linestyle=linestyle[key])
ax.legend()
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

# Print table of summary data re: accuracy
table_header = ['Output',
                # 'n_nodes', 'depth',
                # 'Train ROC AUC', 'Test ROC AUC',
                'Manual',
                '20 trees\n(5CV)',
                '20 trees\n(nCV)',
                '1000 trees\n(5CV)',
                '1000 trees\n(nCV)',
                '20 trees\nno control\n(5CV)',
                '20 trees\nno control\n(nCV)',
                '1000 trees\nno control\n(5CV)',
                '1000 trees\nno control\n(nCV)',
                'Score',
                'OOB']
table_content = list()
for key in parameters:
    table_content.append([key,
                          # np.mean(n_nodes[key]), np.mean(depth[key]),
                          # train_roc_auc[key], test_roc_auc[key],
                          accuracy[key],
                          "{:0.2f} ({:0.2f})".format(cv5_scores20[key].mean()*100, cv5_scores20[key].std()*100),
                          "{:0.2f} ({:0.2f})".format(cvn_scores20[key].mean()*100, cvn_scores20[key].std()*100),
                          "{:0.2f} ({:0.2f})".format(cv5_scores1000[key].mean()*100, cv5_scores1000[key].std()*100),
                          "{:0.2f} ({:0.2f})".format(cvn_scores1000[key].mean()*100, cvn_scores1000[key].std()*100),
                          "{:0.2f} ({:0.2f})".format(cv5_scores20_nocontrol[key].mean()*100,
                                                     cv5_scores20_nocontrol[key].std()*100),
                          "{:0.2f} ({:0.2f})".format(cvn_scores20_nocontrol[key].mean()*100,
                                                     cvn_scores20_nocontrol[key].std()*100),
                          "{:0.2f} ({:0.2f})".format(cv5_scores1000_nocontrol[key].mean()*100,
                                                     cv5_scores1000_nocontrol[key].std()*100),
                          "{:0.2f} ({:0.2f})".format(cvn_scores1000_nocontrol[key].mean()*100,
                                                     cvn_scores1000_nocontrol[key].std()*100),
                          rf[key].score(test_features[key], test_labels[key]),
                          oob_accuracy[key]])
print("ACCURACY TABLES (mean +/- std)")
print("==============================")
print(tabulate(table_content, headers=table_header, tablefmt='simple'))
print("")

# Calculate relative importances of metrics in decision trees
# feature_importances = dict.fromkeys(parameters)
feature_importances = pd.DataFrame(index=parameters, columns=metrics)
feature_std = pd.DataFrame(index=parameters, columns=metrics)
for key in parameters:
    importances = list(rf[key].feature_importances_)
    std = np.std([tree.feature_importances_ for tree in rf[key].estimators_], axis=0)
    i_metric = 0
    for met_key in metrics:
        feature_importances.loc[key, met_key] = round(importances[i_metric], 2)
        feature_std.loc[key, met_key] = round(std[i_metric], 2)
        i_metric += 1
# feature_importances[key] = [[feature, round(importance, 2), round(std, 2)] for feature, importance, std in zip(
#     df_metrics.columns, importances, std)]
# Sort the feature importances by most important first
# feature_importances[key] = sorted(feature_importances[key], key = lambda x: x[1], reverse = True)

# Print table of relative importance of each metric for each parameter
print(tabulate(feature_importances, headers='keys', tablefmt='psql'))

# Plot a series of bar graphs to demonstrate the relative importances of metrics (potentially only the important ones!)
params_predict = ['lv', 'septum']
importances_predict = feature_importances.loc[params_predict, :].astype(float)
mean_val = importances_predict.describe().loc['mean', :]
mean_val_sort, metric_import_sort, importances_predict_sort = zip(*sorted(zip(mean_val, metrics, importances_predict)))
fig, ax = plt.subplots()
x = np.arange(len(metrics))
bar_width = 0.9 / len(params_predict)
start_x = x - (bar_width * (len(params_predict) / 2))
xmin = min(start_x)
for key in params_predict:
    # importances = np.array(feature_importances.loc[key, :])
    # ax.bar(start_x, importances, bar_width, label=key, align='edge')
    importances = np.array(importances_predict.loc[key, list(metric_import_sort)])
    ax.bar(start_x, importances, bar_width, label=key, align='edge')
    start_x += bar_width
xmax = max(start_x)
ax.set_ylabel('Relative importance')
ax.set_xticks(x)
# ax.set_xticklabels(metrics, rotation=45)
ax.set_xticklabels(metric_import_sort, rotation=45)
ax.set_xlim([xmin, xmax])
ax.legend()
vals = ax.get_yticks()
if vals[-1] > ax.get_ylim()[-1]:
    vals = np.delete(vals, -1)
for val in vals:
    ax.axhline(y=val, linestyle='dashed', color='k', zorder=-1)
fig.tight_layout()

# Print details of LV/septum predictions in more detail
table_header = ['LV prob', 'Sept prob', 'LV Pred', 'Sept Pred', 'Actual']
table_content = list()
for i in range(len(test_labels['lv'])):
    if predictions['lv'][i] == 1:
        lv_pred_text = 'lv'
    else:
        lv_pred_text = 'None'
    if predictions['septum'][i] == 1:
        sept_pred_text = 'septum'
    else:
        sept_pred_text = 'None'

    if test_labels['lv'][i] == 1:
        act_text = 'lv'
    elif test_labels['septum'][i] == 1:
        act_text = 'septum'
    else:
        act_text = 'None'
    table_content.append([probabilities['lv'][i], probabilities['septum'][i], lv_pred_text, sept_pred_text, act_text])
print(tabulate(table_content, headers=table_header, tablefmt='simple'))

# Conduct visualisations
for key in parameters:
    # Pull out one tree from the forest
    # tree = rf[key].estimators_[5] # Pull out one tree from the forest
    for i, tree in enumerate(rf[key].estimators_):
        # Export the image to a dot file
        export_graphviz(tree, out_file='tree.dot', rounded=True, filled=True, precision=2, feature_names=metrics)
        # Use dot file to create a graph
        (graph,) = pydot.graph_from_dot_file('tree.dot')
        # Write graph to a png file
        graph.write_png('tree' + str(key) + '_' + str(i) + '.png')
        if i > 10:
            break

# Calculate AUC analysis for whether each "important" metric can predict LV/septum. For the basis of this,
# we shall ask the question whether it is LV or not (not whether it is septum or not)
# metrics_important = ['qrs_duration', 'qrs_area_pythag', 'maxMag', 'dTuwd', 'dTmax', 'dTqrs100']
metrics_important = metrics.copy()
# Correct potential issues with data
for key in metrics_important:
    for i in df_metrics_nocontrol.loc[:, 'maxMagTime'].index:
        if isinstance(df_metrics_nocontrol.loc[i, 'maxMagTime'], list) or \
                isinstance(df_metrics_nocontrol.loc[i, 'maxMagTime'], np.ndarray):
            df_metrics_nocontrol.loc[i, 'maxMagTime'] = df_metrics_nocontrol.loc[i, 'maxMagTime'][0]
df_full_nocontrol = pd.concat([df_metrics_nocontrol, df_params_nocontrol], axis=1)
tpr_manual = dict.fromkeys(metrics_important)
fpr_manual = dict.fromkeys(metrics_important)
tp = dict.fromkeys(metrics_important)
fp = dict.fromkeys(metrics_important)
tn = dict.fromkeys(metrics_important)
fn = dict.fromkeys(metrics_important)
for key in metrics_important:
    # Search over each value recorded of the metric
    metric_vals = sorted(set(df_metrics_nocontrol.loc[:, key]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    tpr_manual[key] = [0]
    fpr_manual[key] = [0]
    tp[key] = [0]
    fp[key] = [0]
    tn[key] = [0]
    fn[key] = [0]
    for metric_val in metric_vals:
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for _, row in df_full_nocontrol.iterrows():
            if row[key] <= metric_val:
                if row['lv'] == 1:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if row['lv'] == 1:
                    false_negatives += 1
                else:
                    true_negatives += 1
        tp[key].append(true_positives)
        fp[key].append(false_positives)
        tn[key].append(true_negatives)
        fn[key].append(false_negatives)
        tpr_temp = true_positives/(true_positives+false_negatives)
        fpr_temp = false_positives/(false_positives+true_negatives)
        assert tpr_temp >= tpr_manual[key][-1], "Failed for {}".format(key)
        assert fpr_temp >= fpr_manual[key][-1], "Failed for {}".format(key)
        tpr_manual[key].append(tpr_temp)
        fpr_manual[key].append(fpr_temp)

auc = dict.fromkeys(metrics_important)
table_content = list()
table_header = ['Metric', 'AUC']
for key in metrics_important:
    auc[key] = np.trapz(tpr_manual[key], x=fpr_manual[key])
    table_content.append([key, auc[key]])
print(tabulate(table_content, headers=table_header, tablefmt='simple'))
print("")

for key in metrics_important:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fpr_manual[key], tpr_manual[key], linewidth=3, linestyle='-', marker='.', markersize=12)
    ax.plot([0, 1], [0, 1], linestyle=':', color='r')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(key)
