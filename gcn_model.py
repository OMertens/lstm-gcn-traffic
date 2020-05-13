import pandas as pd
import numpy as np
from numpy import array
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ax import optimize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import time
import math

from sklearn.utils import shuffle
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance


# TABLE OF CONTENTS:
# 1. Build DataFrame (line 47)
# 2. Drop unreliable detectors (line 64)
# 3. Roll-up and cleaning (line 122)
# 4. Edge weight calculation (line 227)
# 5. Data pre-processing (line 492)
# 6. GCN architecture (line 668)
# 7. Functions for train / test (line 704)
# 8. Plot (line 736)
# 9. Instantiate and run model (line 751)
# 10. Hyper parameter tuning (line 910)


# check for cuda availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# assign fixed seed for reproducibility
np.random.seed(123)
torch.manual_seed(123)

# Load dataset and transform to pandas DataFrame
pickle = pd.read_pickle("brussels_traffic_counting.pkl")
df = pd.DataFrame(pickle)


# ----------------------------------------------------------------------------------------------------
# 1. BUILD DATAFRAME
# ----------------------------------------------------------------------------------------------------

# Change columns 'count', 'speed' and 'occupancy' to floats
df[['count', 'speed', 'occupancy']] = df[['count', 'speed', 'occupancy']].apply(pd.to_numeric)

# Change columns 'start_time' and 'end_time' to pandas datetime
df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce', format='%Y/%m/%d %H:%M')
df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce', format='%Y/%m/%d %H:%M')

# Add 'hour', 'date' and 'week_day' column
df['hour'] = df['start_time'].dt.hour  # hour (0 - 23)
df['date'] = df['start_time'].dt.date  # name of the day: e.g. Monday, Tuesday, ...
df['week_day'] = df['start_time'].dt.dayofweek  # day as numerical (0 = Monday, 6 = Sunday)


# ----------------------------------------------------------------------------------------------------
# 2. DROP UNRELIABLE DETECTORS
# ----------------------------------------------------------------------------------------------------

def count_nulls_per_detector():  # NOT USED AT THE MOMENT
    print(df.detector_code.unique())

    for k, i in enumerate(df.detector_code.unique()):
        print("DETECTOR CODE: " + i + ". DETECTOR NUMBER: " + str(k))
        print(df[df['detector_code'] == i].isnull().sum())


def modify_row(data, id, count, speed, occupancy):  # NOT USED AT THE MOMENT
    data.loc[id, 'count'] = count
    data.loc[id, 'speed'] = speed
    data.loc[id, 'occupancy'] = occupancy


def get_detector_df(data, detector):
    # slice DataFrame on a single detector
    df_detector = data[data['detector_code'] == detector]
    df_detector = df_detector.reset_index()
    return df_detector


def detectors_to_keep(data):
    all_detectors = data.detector_code.unique()
    # drop 0-19% completeness detectors
    detectors_to_drop = ['SB0236_BHout', 'SB0246_BAout', 'SB121_BBin', 'SB125_BBin', 'SB125_BBout', 'SGN02_BAout',
                         'SUL62_BA1out', 'SUL62_BDin', 'SUL62_BDout', 'SUL62_BGin', 'SUL62_BHin', 'SUL62_BHout']
    # drop 99% completeness detectors
    detectors_to_drop += ['SB0236_BCout', 'SB1201_BAout']
    # drop near 100% completeness detectors
    detectors_to_drop += ['SB020_BAout', 'SB020_BBin', 'SB020_BCin', 'SB020_BDout']
    det_to_keep = []
    for det in all_detectors:
        if det not in detectors_to_drop:
            det_to_keep.append(det)
    return det_to_keep  # list of detector codes


def list_df_detectors(data, det_list):
    df_det_list = []
    for det in det_list:
        df_det = get_detector_df(data, det)
        df_det_list.append(df_det)
    return df_det_list  # list of detector DataFrames


detectors = detectors_to_keep(df)
detectors_df_list = list_df_detectors(df, detectors)
df = pd.concat(detectors_df_list)
df = df.reset_index()
df = df.drop(['level_0', 'index'], axis=1)

# df is now DataFrame with all detectors without null values (49 detectors)


# ----------------------------------------------------------------------------------------------------
# 3. ROLL UP AND CLEANING
# ----------------------------------------------------------------------------------------------------

def roll_up(input_detector):
    # Roll up to 5 min interval
    roll_up_detector = input_detector.set_index('start_time').resample('5T').mean()
    return roll_up_detector


def gaps_check(input_detector):
    # Check for gaps in data
    # df_detector_null = input_detector[input_detector['count'].isnull()]
    # print(df_detector_null.index.unique())

    # CLEAN SMALL GAPS (5 - 15 min)
    input_detector = input_detector.ffill(axis=0)
    gaps_detector = input_detector.reset_index()
    # CLEAN 6 HOUR GAP ON JAN 28TH
    index = 5061
    num_missing_values = 70
    gaps_detector = fill_by_similar_day(gaps_detector, index, num_missing_values)
    gaps_detector = gaps_detector.drop(['index'], axis=1)

    return gaps_detector


def fill_by_similar_day(df, index, num_missing_values):
    # Used to fill the 6 hour gap on Jan 28th.
    # Filled with average of Jan 21st and Jan 14th.
    # 1 week is 2016 five-minute intervals.
    for i in range(num_missing_values):
        df.loc[index + i, 'count'] = (df.loc[(index + i) - 2016, 'count'] + df.loc[
            (index + i) - (2016 * 2), 'count']) / 2
        df.loc[index + i, 'speed'] = (df.loc[(index + i) - 2016, 'speed'] + df.loc[
            (index + i) - (2016 * 2), 'speed']) / 2
        df.loc[index + i, 'occupancy'] = (df.loc[(index + i) - 2016, 'occupancy'] + df.loc[
            (index + i) - (2016 * 2), 'occupancy']) / 2
    return df


def create_speed_variance(input_detector):
    # Speed variance for at time t is the variance of speed values between t - 5 and t
    # The first 5 values (out of 6000+) are filled with 10's, as this seemed to be close to the mean variance.
    input_detector_speed_list = input_detector['speed'].tolist()
    input_detector_variance_last_items = get_variance(input_detector_speed_list)
    first_items = [10, 10, 10, 10, 10]
    input_detector_variance = first_items + input_detector_variance_last_items

    input_detector['speed_variance'] = input_detector_variance
    return input_detector


def get_variance(list):
    variance_list = []
    for i in range(len(list) - 5):
        variance = np.var(list[i: i + 5])
        variance_list.append(variance)
    return variance_list


def get_clean_detector(input_detector):
    # Add all previous functions in 1 function to get a clean, rolled-up DataFrame
    # of a single detector.
    input_detector_df = get_detector_df(df, input_detector)
    input_detector_df = roll_up(input_detector_df)
    input_detector_df = gaps_check(input_detector_df)
    input_detector_df = create_speed_variance(input_detector_df)
    return input_detector_df


# List of detector codes which are used in the graph
input_detector_codes = ['ARL_103',
                        'ARL_203',
                        'BOT_TD2',
                        'HAL_191',
                        'HAL_292',
                        'LOU_110',
                        'LOU_TD1',
                        'LOU_TD2',
                        'MAD_103',
                        'MAD_203',
                        'PNA_103',
                        'PNA_203',
                        'ROG_TD1',
                        'ROG_TD2',
                        'STE_TD1',
                        'STE_TD2',
                        'STE_TD3',
                        'TRO_203',
                        'TRO_TD1',
                        'TRO_TD2']

input_detectors_list = []
for code in input_detector_codes:
    clean_det_df = get_clean_detector(code)
    input_detectors_list.append(clean_det_df)

# As of here: input_detectors_list is a list of DataFrames of all detectors being used in the graph

# PNA_203 contains one negative occupancy and count value on index 700, set to zero
input_detectors_list[11].loc[700, 'occupancy'] = 0
input_detectors_list[11].loc[700, 'count'] = 0


# ----------------------------------------------------------------------------------------------------
# 4. EDGE WEIGHT CALCULATION
# ----------------------------------------------------------------------------------------------------

# List of coordinates (x,y) detectors
coor = [(149983.844584616,170594.458471007),
      (149980.625675607,170557.307570162),
      (149880.7038,171392.2744),
      (148330.07,169206.0073),
      (148281.1008,169212.453),
      (149010.2063,169466.4362),
      (149169.2506,169575.1253),
      (149051.9216,169486.5036),
      (150015.439227949,170972.542788558),
      (150029.277301607,170939.083927821),
      (149550.9689784,169828.02599149),
      (149503.135301163,169785.17581533),
      (149052.6073,171744.8667),
      (149069.6816,171744.8277),
      (149054.988357091,169465.007444001),
      (149062.389857091,169441.974944001),
      (149067.227557091,169478.620644001),
      (149798.454027391,170027.226854425),
      (149825.369929864,170123.502967118),
      (149819.158567755,170083.129113408)
      ]

# Orientation of detectors
orientation = [20, 200, 120, 110, 290, 105, 50, 230, 350, 170, 55, 235, 295, 115, 140, 325, 190, 220, 20, 200]


# Make list of indexes of adjacent node
def search_node_index(list):
    all_index =[]
    for i in range(len(list)):
        if list[i]:
            all_index.append(i)
    return all_index


# Make list of adjacent nodes by using indexes
def create_node_list_on_index(index_list, node_combo_list):
    all_nodes= []
    for index in index_list:
        node = node_combo_list[index]
        all_nodes.append(node)
    return all_nodes


# Normalize
def normalization(weight_list):
    normalized_list = []
    for weight in weight_list:
        min_of_list = min(weight_list)
        max_of_list = max(weight_list)
        norm_weight = (weight - min_of_list)/(max_of_list-min_of_list)
        normalized_list.append(norm_weight)

    return normalized_list


# Calculate orientation similarity
def calc_orient_similarity(edge_list,orientation_list):
    orient_feature_list= []
    for edge in edge_list:
        x = int(edge[0])
        y = int(edge[1])
        orient_feature = math.cos(abs(orientation_list[x]-orientation_list[y]))
        # Set values between 0 and 1
        orient_feature = (orient_feature+1) /2
        orient_feature_list.append(orient_feature)
    return orient_feature_list


# Calculate spatial similarity
def calc_spatial_similarity(edge_list,coor_list):
    spatial_feature_list= []
    for edge in edge_list:
        x = int(edge[0])
        y = int(edge[1])
        coor_node1 = coor_list[x]
        coor_node2 = coor_list[y]
        spatial_feature = distance.euclidean(coor_node1,coor_node2)
        spatial_feature_list.append(spatial_feature)

    return spatial_feature_list


# Calculate euclidean weights
def calc_eucl_weights(index_list, eucl_list):
    weights_list = []
    for index in index_list:
            weight = 1 - eucl_list[index]
            weights_list.append(weight)
    return weights_list


# Calculate travel distance weights
def calc_travel_weights(index_list, travel_list):
    weights_list = []
    for index in index_list:
            weight = 1 - travel_list[index]
            weights_list.append(weight)
    return weights_list


# Calculate correlation weights
def calc_corr_weights(index_list, correlation_list):
    weights_list = []
    for index in index_list:
        weight = correlation_list[index]
        weights_list.append(weight)
    return weights_list


# Calculate euclidean + orientation similarity
def calc_eucl_orient_similarity(spatial_list, orient_list):
    weights_list = []
    for i in range(len(spatial_list)):
        calculated_weight = (orient_list[i]+ (1-spatial_list[i]))/2
        weights_list.append(calculated_weight)
    return weights_list


# Calculate euclidean + orientation weights
def calc_eucl_orient_weights(index_list, similarity_list):
    weights_list = []
    for index in index_list:
        weight = similarity_list[index]
        weights_list.append(weight)
    return weights_list


# Calculation for euclidean graph
def euclidean_graph(threshold):
    # 1) DETERMINE EDGE INDEX

    # Import node combinations
    node_combos = pd.read_csv('node_combination.csv', header=None)
    node_combos = array(node_combos)
    # Import normalized euclidean distances of the node combinations
    distances = pd.read_csv('edges_weight_euclidean.csv', header = None)
    distances = array(distances)

    # Calculate which nodes are adjacent according to threshold
    dist_boolean = distances < threshold
    index_list = search_node_index(dist_boolean)
    edge_index_list = create_node_list_on_index(index_list, node_combos)

    # 2) CALCULATE EDGE WEIGHTS

    edge_weights = calc_eucl_weights(index_list, distances)

    for i in range(len(edge_weights)):
        edge_weights[i] = float(edge_weights[i])

    for i in range(len(edge_index_list)):
        edge_index_list[i] = list(edge_index_list[i])
        for j in range(len(edge_index_list[i])):
            edge_index_list[i][j] = int(edge_index_list[i][j])

    return edge_index_list, edge_weights


# Calculation for travel distance graph
def travel_graph(threshold):
    # 1) DETERMINE EDGE INDEX

    # Import node combinations
    node_combos = pd.read_csv('node_combination.csv', header=None)
    node_combos = array(node_combos)
    # Import normalized travel distances of the node combinations
    distances = pd.read_csv('edges_weight_travel.csv', header=None)
    distances = array(distances)

    # Calculate which nodes are adjacent according to threshold
    dist_boolean = distances < threshold
    index_list = search_node_index(dist_boolean)
    edge_index_list = create_node_list_on_index(index_list, node_combos)

    # 2) CALCULATE EDGE WEIGHTS

    edge_weights = calc_travel_weights(index_list, distances)

    for i in range(len(edge_weights)):
        edge_weights[i] = float(edge_weights[i])

    for i in range(len(edge_index_list)):
        edge_index_list[i] = list(edge_index_list[i])
        for j in range(len(edge_index_list[i])):
            edge_index_list[i][j] = int(edge_index_list[i][j])

    return edge_index_list, edge_weights


# Calculation for correlation graph
def correlation_graph(threshold):
    # 1) DETERMINE EDGE INDEX

    # Import node combinations
    node_combos = pd.read_csv('node_combination.csv', header=None)
    node_combos = array(node_combos)
    # Import travel distances of the node combinations
    correlations = pd.read_csv('edges_weight_correlation.csv', header=None)
    correlations = array(correlations)

    # Calculate which nodes are adjacent according to threshold
    corr_boolean = correlations > threshold
    index_list = search_node_index(corr_boolean)
    edge_index_list = create_node_list_on_index(index_list,node_combos)

    # 2) CALCULATE EDGE WEIGHTS

    edge_weights = calc_corr_weights(index_list,correlations)

    for i in range(len(edge_weights)):
        edge_weights[i] = float(edge_weights[i])

    for i in range(len(edge_index_list)):
        edge_index_list[i] = list(edge_index_list[i])
        for j in range(len(edge_index_list[i])):
            edge_index_list[i][j] = int(edge_index_list[i][j])

    return edge_index_list, edge_weights


# Calculation for euclidean + oriental graph
def eucl_orient_graph (coordinates,orientation, threshold):
    # 1) DETERMINE EDGE INDEX

    # Import node combinations
    node_combos = pd.read_csv('node_combination.csv', header=None)
    node_combos = array(node_combos)

    # Calculate orientation similarity between nodes
    orientations = calc_orient_similarity(node_combos, orientation)
    norm_orientations = normalization(orientations)

    # Calculate spatial similarity between nodes
    eucl_distances = calc_spatial_similarity(node_combos, coordinates)
    norm_eucl_distances = normalization(eucl_distances)

    # Calculate which nodes are adjacent according to threshold
    similarity = calc_eucl_orient_similarity(norm_eucl_distances, norm_orientations)
    similarity = array(similarity)
    #print(similarity)
    sim_boolean = similarity > threshold
    #print(sim_boolean)
    index_list = search_node_index(sim_boolean)
    edge_index_list = create_node_list_on_index(index_list, node_combos)
    #print(edge_index_list)
    # 2) CALCULATE EDGE WEIGHTS

    edge_weights = calc_eucl_orient_weights(index_list,similarity)

    for i in range(len(edge_weights)):
        edge_weights[i] = float(edge_weights[i])

    for i in range(len(edge_index_list)):
        edge_index_list[i] = list(edge_index_list[i])
        for j in range(len(edge_index_list[i])):
            edge_index_list[i][j] = int(edge_index_list[i][j])

    return edge_index_list, edge_weights

# ----------------------------------------------------------------------------------------------------
# 5. DATA PRE-PROCESSING
# ----------------------------------------------------------------------------------------------------

def norm_fit_transform(df_col, scale):
    # function for normalizing sequence by min-max scaling
    seq = array(df_col)
    seq = scale.fit_transform(seq.reshape(-1, 1))
    seq = seq.ravel()
    return seq


def norm_transform(df_col, scale):
    # function for normalizing sequence on the scale of the last fit_transform
    seq = array(df_col)
    seq = scale.transform(seq.reshape(-1, 1))
    seq = seq.ravel()
    return seq


def get_list_of_samples(detector_list, n_steps):
    # Data pre-processing resulting in X_train, y_train, X_test, y_test
    # Methodology:
    # All DataFrames of detectors in the graph are used as input. Every detector DataFrame is split in train and test.
    # Then the train set of every detector is concatenated into one large train DataFrame.
    # The same happens for test set.
    # Now that all train occupancy, count, ... are in a single DataFrame, MinMaxScaler can be used to fit_transform.
    # Then MinMaxScaler is used to transform on test data set.
    # With this approach, test data is normalized on the scale of train data and can be slightly above 1 or slightly
    # below zero. This is compliant with best practices. If we scale on the entire data set at once, the train set
    # will be slightly biased on the test set already.

    # After having normalized the train and test DataFrame, these are split up again in smaller train and test
    # DataFrames for each detector. Then samples are made from these DataFrames.

    # A sample might look like this:
    # SAMPLE: [det1 occupancy t-1, det1 speed t-1, det1 count t-1], [det2 occupancy t-1, det2 speed t-1, det2 count t-2]
    # LABEL: [det1 occupancy t], [det2 occupancy t]
    # X_train and X_test are lists of samples
    # y_train and y_test are lists of labels

    length = len(detector_list[0]['occupancy'])
    train_length = int(length * 0.8)
    test_length = length - train_length

    # These booleans can be changed to include extra features. Occupancy is always included.
    include_count = True
    include_speed = False
    include_speed_var = True
    include_weekday = True
    normalized_target = True

    train_dets, test_dets = list(), list()
    for detector in detector_list:
        train_per_det = detector[:train_length]
        test_per_det = detector[train_length:]
        train_dets.append(train_per_det)
        test_dets.append(test_per_det)

    unnorm_occupancy_cols_train = []  # len 20, 4945
    for detector in train_dets:
        unnorm_occupancy_cols_train.append(array(detector['occupancy']))

    unnorm_occupancy_cols_test = []  # len 20, 1237
    for detector in test_dets:
        unnorm_occupancy_cols_test.append(array(detector['occupancy']))

    train_df = pd.concat(train_dets)
    minimum_occupancy, maximum_occupancy = min(train_df['occupancy']), max(train_df['occupancy'])
    test_df = pd.concat(test_dets)

    scaler = MinMaxScaler(feature_range=(0, 1))

    train_occ = norm_fit_transform(train_df['occupancy'], scaler)
    test_occ = norm_transform(test_df['occupancy'], scaler)

    train_speed = norm_fit_transform(train_df['speed'], scaler)
    test_speed = norm_transform(test_df['speed'], scaler)

    train_count = norm_fit_transform(train_df['count'], scaler)
    test_count = norm_transform(test_df['count'], scaler)

    train_speed_var = norm_fit_transform(train_df['speed_variance'], scaler)
    test_speed_var = norm_transform(test_df['speed_variance'], scaler)

    train_weekday = norm_fit_transform(train_df['week_day'], scaler)
    test_weekday = norm_transform(test_df['week_day'], scaler)

    train_feats = [train_occ, train_speed, train_count, train_speed_var, train_weekday]
    test_feats = [test_occ, test_speed, test_count, test_speed_var, test_weekday]

    detectors_train, detectors_test = list(), list()
    for i in range(len(detector_list)):
        detector_train_feats = []
        for feat in train_feats:
            det_train_feat = list(feat[train_length * i:train_length * (i + 1)])
            detector_train_feats.append(det_train_feat)
        detectors_train.append(detector_train_feats)

        detector_test_feats = []
        for feat in test_feats:
            det_test_feat = list(feat[test_length * i:test_length * (i + 1)])
            detector_test_feats.append(det_test_feat)
        detectors_test.append(detector_test_feats)

    X_train, y_train = list(), list()
    for i in range(train_length):
        end_ix = i + n_steps

        if end_ix >= train_length:
            break

        train_sample, train_label = list(), list()

        for k, detector in enumerate(detectors_train):
            if normalized_target:
                det_y = detector[0][end_ix]
            else:
                det_y = unnorm_occupancy_cols_train[k][end_ix]
            det_occupancy = list(detector[0][i:end_ix])
            det_speed = list(detector[1][i:end_ix])
            det_count = list(detector[2][i:end_ix])
            det_speed_var = list(detector[3][i:end_ix])
            det_weekday = list(detector[4][i:end_ix])
            det_x = det_occupancy
            if include_count:
                det_x += det_count
            if include_speed:
                det_x += det_speed
            if include_speed_var:
                det_x += det_speed_var
            if include_weekday:
                det_x += det_weekday
            train_sample.append(det_x)
            train_label.append(det_y)

        X_train.append(train_sample)
        y_train.append(train_label)

    X_test, y_test = list(), list()
    for i in range(test_length):
        end_ix = i + n_steps

        if end_ix >= test_length:
            break

        test_sample, test_label = list(), list()

        for k, detector in enumerate(detectors_test):
            if normalized_target:
                det_y = detector[0][end_ix]
            else:
                det_y = unnorm_occupancy_cols_test[k][end_ix]
            det_occupancy = list(detector[0][i:end_ix])
            det_speed = list(detector[1][i:end_ix])
            det_count = list(detector[2][i:end_ix])
            det_speed_var = list(detector[3][i:end_ix])
            det_weekday = list(detector[4][i:end_ix])
            det_x = det_occupancy
            if include_count:
                det_x += det_count
            if include_speed:
                det_x += det_speed
            if include_speed_var:
                det_x += det_speed_var
            if include_weekday:
                det_x += det_weekday
            test_sample.append(det_x)
            test_label.append(det_y)

        X_test.append(test_sample)
        y_test.append(test_label)

    return X_train, y_train, X_test, y_test, minimum_occupancy, maximum_occupancy


# ----------------------------------------------------------------------------------------------------
# 6. GCN ARCHITECTURE
# ----------------------------------------------------------------------------------------------------

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_gcn_layers, add_dropout, n_nodes1, n_nodes2):
        super(GCN, self).__init__()
        self.num_gcn_layers = num_gcn_layers
        self.add_dropout = add_dropout

        # 2 convolutional layers, 1 linear layer
        self.conv1 = GCNConv(num_node_features, n_nodes1)
        if self.num_gcn_layers == 2:
            self.conv2 = GCNConv(n_nodes1, n_nodes2)
            self.lin1 = torch.nn.Linear(n_nodes2, 1)
        else:
            self.lin1 = torch.nn.Linear(n_nodes1, 1)

    def forward(self, data, weights_matrix):
        x, edge_index = data.x, data.edge_index

        # Activation function elu, very often used for GCNs.
        x = self.conv1.forward(x, edge_index, weights_matrix)
        x = F.elu(x)
        if self.add_dropout:
            x = F.dropout(x, training=self.training)
        if self.num_gcn_layers == 2:
            x = self.conv2.forward(x, edge_index, weights_matrix)
            x = F.elu(x)
            if self.add_dropout:
                x = F.dropout(x, training=self.training)
        x = self.lin1.forward(x)

        return x.view(x.shape[0])


# ----------------------------------------------------------------------------------------------------
# 7. FUNCTIONS FOR TRAIN / TEST
# ----------------------------------------------------------------------------------------------------

def train_on_graph(input_graph, train_edge_weights, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    # out is a tensor of size 20, one prediction for each node.
    out = model(input_graph, train_edge_weights)

    single_train_loss = criterion(out, input_graph.y)

    # print denormalized loss per sample for testing:
    # print("SAMPLE LOSS: ", (single_train_loss.item() * (max_occupancy - min_occupancy)) + min_occupancy)

    single_train_loss.backward()
    optimizer.step()

    return single_train_loss


def test_on_graph(input_graph, train_edge_weights, model, criterion):
    model.eval()

    out = model(input_graph, train_edge_weights)

    single_test_loss = criterion(out, input_graph.y)

    return single_test_loss


# ----------------------------------------------------------------------------------------------------
# 8. PLOT
# ----------------------------------------------------------------------------------------------------


def plot_results(train_loss_list, test_loss_list):
    plt.title("Denormalized MAE Loss train/test")
    plt.xlabel("Epoch")
    plt.ylabel("Denormalized MAE Loss")
    plt.plot(train_loss_list, label="train loss")
    plt.plot(test_loss_list, label="test loss")
    plt.legend()
    plt.show()


# ----------------------------------------------------------------------------------------------------
# 9. INSTANTIATE AND RUN MODEL
# ----------------------------------------------------------------------------------------------------

def run_gcn(p):

    epochs = 1000
    plot = True

    print('Parameters:')
    print('timesteps:', p['timesteps'])
    print('n_gcn_layers:', p['n_gcn_layers'])
    print('num_nodes_1', p['nodes1'])
    print('num_nodes_2', p['nodes2'])
    print('weight decay:', p['weight_decay'])
    print('dropout:', p['dropout'])
    print('threshold:', p['threshold'])

    def denormalize(x):
        return (x * (max_occupancy - min_occupancy)) + min_occupancy

    # euclidean_graph (threshold) smaller threshold, smaller graph
    # travel_graph (threshold) smaller threshold, smaller graph
    # correlation_graph (threshold) higher threshold, smaller graph
    # eucl_orient_graph (coor, orientation, threshold) higher threshold, smaller graph

    edge_index_as_list, weights_as_list = euclidean_graph(p['threshold'])

    print("edge index length:")
    print(len(edge_index_as_list))

    edge_index = torch.tensor(edge_index_as_list, dtype=torch.long)
    edge_weights = torch.tensor(weights_as_list, device=device)

    # Lists of samples and labels
    # Number of time steps is chosen here as second argument.
    # Because of the way temporal dependencies are currently implemented,
    # number of node features = (number for features) * (number of time steps)
    X_train, y_train, X_test, y_test, min_occupancy, max_occupancy = get_list_of_samples(input_detectors_list, p['timesteps'])

    # Shuffle train set
    X_train = array(X_train)
    y_train = array(y_train)
    X_train, y_train = shuffle(X_train, y_train)
    X_train, y_train = list(X_train), list(y_train)

    # Our samples are lists and still need to be transformed to tensors and then put into graphs.
    train_graphs = list()
    for i in range(len(X_train)):
        x_tensor = torch.tensor(X_train[i], dtype=torch.float)
        y_tensor = torch.tensor(y_train[i], dtype=torch.float)
        sample = Data(x=x_tensor, edge_index=edge_index.t().contiguous(), y=y_tensor)
        train_graphs.append(sample)

    test_graphs = list()
    for i in range(len(X_test)):
        x_tensor = torch.tensor(X_test[i], dtype=torch.float)
        y_tensor = torch.tensor(y_test[i], dtype=torch.float)
        sample = Data(x=x_tensor, edge_index=edge_index.t().contiguous(), y=y_tensor)
        test_graphs.append(sample)

    # As of here, train_graphs and test_graphs are lists of 4,945 and 1,237 graphs respectively.
    # train_graphs can be interpreted as X_train and y_train combined and the edge index added, same goes for test_graphs.

    model = GCN(train_graphs[0].num_node_features, p['n_gcn_layers'], p['dropout'], p['nodes1'], p['nodes2']).to(device)

    criterion = nn.L1Loss()  # L1Loss is Mean Absolute Error in PyTorch.
    if p['weight_decay']:
        optimizer = torch.optim.Adam(model.parameters(), lr=p['lr'], weight_decay=5e-4)  # weight decay removed for now.
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=p['lr'])
    # weight_decay=5e-4

    # assign train_graphs and test_graphs to cuda / cpu
    train_data = []
    for graph in train_graphs:
        graph_on_device = graph.to(device)
        train_data.append(graph_on_device)

    test_data = []
    for graph in test_graphs:
        graph_on_device = graph.to(device)
        test_data.append(graph_on_device)

    train_loss_list, test_loss_list = list(), list()
    avg_train_loss_10ep_list, avg_test_loss_10ep_list = list(), list()

    # single loss list used for testing:
    # single_train_loss_list, single_test_loss_list = list(), list()

    print("Start training...")

    t0 = time.time()  # 45 - 55 seconds per epoch on cuda enabled GTX 1660 Ti
    for epoch in range(epochs):
        print("Epoch: ", epoch)

        # Loss is calculated for each sample for backpropagation, but average loss is calculated for
        # each epoch for visualization.

        train_loss = 0
        for k, graph in enumerate(train_data):
            train_loss += denormalize(train_on_graph(graph, edge_weights, model, criterion, optimizer).item())
            # Used for testing:
            # single_train_loss_list.append(train_on_graph(graph, edge_weights, model, criterion, optimizer).item())
        train_loss /= len(train_data)

        test_loss = 0
        for k, graph in enumerate(test_data):
            with torch.no_grad():
                test_loss += denormalize(test_on_graph(graph, edge_weights, model, criterion).item())
                # Used for testing:
                # single_test_loss_list.append(test_on_graph(graph, edge_weights, model, criterion).item())
        test_loss /= len(test_data)

        print("Denormalized train loss (MAE): ", train_loss)
        print("Denormalized test loss (MAE): ", test_loss)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        # lists for plotting
        if epoch % 10 == 0:
            print()
            print("LENGTH ; ------------")
            print(len(test_loss_list))
            print(test_loss_list)
            avg_train_loss_10ep = np.average(train_loss_list)
            avg_test_loss_10ep = np.average(test_loss_list)
            print(avg_test_loss_10ep)

            avg_train_loss_10ep_list.append(avg_train_loss_10ep)
            avg_test_loss_10ep_list.append(avg_test_loss_10ep)

            train_loss_list = []
            test_loss_list = []
            print("list:", test_loss_list)
            print("END -------")
            print()
        t1 = time.time()
        print(str(int(t1 - t0)) + " seconds for last epoch.")
        t0 = time.time()
        print()

    if plot:
        plot_results(avg_train_loss_10ep_list, avg_test_loss_10ep_list)

    print('Parameters:')
    print('timesteps:', p['timesteps'])
    print('n_gcn_layers:', p['n_gcn_layers'])
    print('num_nodes_1', p['nodes1'])
    print('num_nodes_2', p['nodes2'])
    print('weight decay:', p['weight_decay'])
    print('dropout:', p['dropout'])
    print('threshold:', p['threshold'])
    print("Minimum test loss: ", min(avg_test_loss_10ep_list))

    return min(avg_test_loss_10ep_list)


# ----------------------------------------------------------------------------------------------------
# 10. HYPER PARAMETER TUNING
# ----------------------------------------------------------------------------------------------------

# type can be set to fixed / choice
# next keyword has to be value or values respectively
# fixed takes in a single value for value, choice takes in a list for values

best_parameters, best_values, _, _ = optimize(
    parameters=[{'name': 'timesteps', 'type': 'fixed', 'value': 2},
                {'name': 'n_gcn_layers', 'type': 'fixed', 'value': 2},
                {'name': 'nodes1', 'type': 'fixed', 'value': 128},
                {'name': 'nodes2', 'type': 'fixed', 'value': 64},
                {'name': 'weight_decay', 'type': 'fixed', 'value': False},
                {'name': 'dropout', 'type': 'fixed', 'value': False},
                {'name': 'threshold', 'type': 'fixed', 'value': 0.2},
                {'name': 'lr', 'type': 'fixed', 'value': 0.001}], evaluation_function=run_gcn,
    minimize=True, total_trials=1)
print("best parameters:", best_parameters)
print("best value:", best_values)


