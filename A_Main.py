from B_user_item_maker import UserItemMaker
from C_outlier_checker import OutlierChecker
from D_SLSDR_feature_selector import SLSDR
from F_similarities import SimilarityCalculator
from G_predictor import Predictor
from J_RMSE_computer import RMSEComputer
from L_result_saver import Results
from E_PCA_feature_selector import PCAFeatureSelector
from H_kmeans_clustering import KmeansClustering
from I_Kmeans_predictor import KmeansPredictor
from K_precision_recall_score_computer import PrecisionRecallComputer

################################# Folder Path #################################
# To start we need a folder containing u1.base and u1.test files
# Set the folder_path with your own folder's path
folder_path = '/Users/reza/Programming/old-repositories/Recommender_system_based_on_feature_selection_SLSDR/data_set/'

training_files_list = ['u1_train.csv', 'u2_train.csv', 'u3_train.csv', 'u4_train.csv', 'u5_train.csv']
testing_files_list = ['u1_test.csv', 'u2_test.csv', 'u3_test.csv', 'u4_test.csv', 'u5_test.csv']
sim_methods = ['cosine', 'jaccard', 'pearson', 'adjusted_cosine']
num_features_list = [100, 200, 300, 400, 500, 600]
RMSE_list = []
precision_list = []
recall_list = []
##############################################################################

################################# User-item Matrix Making #################################
# If you don't have the csv files for datasets uncomment the code below
UserItemMaker_agent = UserItemMaker(folder_path)
UserItemMaker_agent.train()
UserItemMaker_agent.test()
##############################################################################

################################# Outlier Checker #################################
OutlierChecker_agent = OutlierChecker(folder_path)
OutlierChecker_agent.training_checker()
OutlierChecker_agent.testing_checker()
##############################################################################

################################# Main Part: SLSDR // Similarity // Prediction // RMSE #################################

for num_feature in num_features_list:
    sim_method = sim_methods[0]

    for i in range(len(training_files_list)):
        # Selecting file names for 5-k fold method
        training_file_name = training_files_list[i]
        testing_file_name = testing_files_list[i]

        # SLSDR feature selection method running over the main training matrices
        SLSDR_feature_selector_agent = SLSDR(file_name=training_file_name, num_feature=num_feature, K=20, steps=100, alpha=0.2, beta=0.1, lambda_value=0.3, sigma=0.1, path=folder_path)
        selected_features_matrix = SLSDR_feature_selector_agent.activator()

        # PCA instead of SLSDR to select important features
        # PCAFeatureSelector_agent = PCAFeatureSelector(path=folder_path, file_name=training_file_name, num_feature=num_feature)
        # selected_features_matrix = PCAFeatureSelector_agent.PCA_runner()

        # Running similarity computing over selected_features_matrix to find similar users
        SimilarityCalculator_agent = SimilarityCalculator(selected_features_matrix)
        similarity_matrix = SimilarityCalculator_agent.get_similarity_matrix(method=sim_method)

        # Predicting the missing rating using collaborative filtering & similarities
        Predictor_agent = Predictor(folder_path, training_file_name, similarity_matrix, 50)
        predicted_rating_matrix = Predictor_agent.weighted_collaborative_filtering()

        # To compute the similarity with k_means comment the codes above (similarity calculator & predictor) and uncomment the codes below

        # Running kmenas_clustering instead of computing similarity
        # KmeansClustering_agent = KmeansClustering(selected_features_matrix, i)
        # labels = KmeansClustering_agent.kmeans_cluster()

        # Predicting the missing rating using collaborative filtering & km_means labels
        # KmeansPredictor_agent = KmeansPredictor(folder_path, training_file_name, labels)
        # predicted_rating_matrix = KmeansPredictor_agent.kmeans_weighted_collaborative_filtering()

        # Computing the RMSE
        RMSEComputer_agent = RMSEComputer(folder_path, testing_file_name, predicted_rating_matrix)
        final_RMSE = RMSEComputer_agent.calculate_RMSE()
        RMSE_list.append(final_RMSE)

        # uncomment if you want to compute the Precision and Recall
        # computing Precision and Recall
        PrecisionRecallComputer_agent = PrecisionRecallComputer(folder_path, testing_file_name, predicted_rating_matrix)
        precision, recall = PrecisionRecallComputer_agent.precision_recall_computer()
        precision_list.append(precision)
        recall_list.append(recall)

    # Compute min/max/avg RMSE
    print(f"RMSE_list = {RMSE_list}")
    min_rmse = min(RMSE_list)
    max_rmse = max(RMSE_list)
    avg_rmse = sum(RMSE_list) / len(RMSE_list)

    # uncomment if you want to compute the Precision and Recall
    # Compute min/max/avg Precision
    min_precision = min(precision_list)
    max_precision = max(precision_list)
    avg_precision = sum(precision_list) / len(precision_list)

    # uncomment if you want to compute the Precision and Recall
    # Compute min/max/avg recall
    min_recall = min(recall_list)
    max_recall = max(recall_list)
    avg_recall = sum(recall_list) / len(recall_list)


##############################################################################


################################# Final Results Saving #################################
    file_name1 = f'{sim_method}_RMSE'
    Results_agent = Results(folder_path, file_name1, num_feature, min_rmse, max_rmse, avg_rmse)
    Results_agent.activator()

    # uncomment if you want to compute the Precision and Recall
    file_name2 = f'{sim_method}_precision'
    Results_agent = Results(folder_path, file_name2, num_feature, min_precision, max_precision, avg_precision)
    Results_agent.activator()

    # uncomment if you want to compute the Precision and Recall
    file_name3 = f'{sim_method}_recall'
    Results_agent = Results(folder_path, file_name3 , num_feature, min_recall, max_recall, avg_recall)
    Results_agent.activator()

##############################################################################
