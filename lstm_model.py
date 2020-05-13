import pandas as pd
import numpy as np
from numpy import array
from numpy import hstack
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from ax import optimize

import torch
import torch.nn as nn
import torch.nn.functional as F

import time


# TABLE OF CONTENTS:
# 1. Build DataFrame (line 43)
# 2. Drop unreliable detectors (line 60)
# 3. Roll-up and cleaning (line 122)
# 4. LSTM pre-processing, sequencing data (line 185)
# 5. LSTM architecture (line 210)
# 6. Functions for train / test (line 266)
# 7. Plotting (line 294)
# 8. Normalisation (line 308)
# 9. Instantiate and run model (line 328)
# 10. Hyper parameter tuning (line 522)


# check for cuda availability
CUDA = torch.cuda.is_available()

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

# Add 'date' and 'hour' column
df['hour'] = df['start_time'].dt.hour
df['date'] = df['start_time'].dt.date
df['week_day'] = df['start_time'].dt.dayofweek


# ----------------------------------------------------------------------------------------------------
# 2. DROP UNRELIABLE DETECTORS
# ----------------------------------------------------------------------------------------------------

def count_nulls_per_detector():
    print(df.detector_code.unique())

    for k, i in enumerate(df.detector_code.unique()):
        print("DETECTOR CODE: " + i + ". DETECTOR NUMBER: " + str(k))
        print(df[df['detector_code'] == i].isnull().sum())


def get_detector_df(data, detector):
    df_detector = data[data['detector_code'] == detector]
    df_detector = df_detector.reset_index()
    return df_detector


def modify_row(data, id, count, speed, occupancy):
    data.loc[id, 'count'] = count
    data.loc[id, 'speed'] = speed
    data.loc[id, 'occupancy'] = occupancy


def clean_negative_count_speed(data):
    for i in data['count']:
        if data[i, 'count'] < 0:
            data[i, 'count'] = 0


def detectors_to_keep(data):
    all_detectors = data.detector_code.unique()
    # drop 0-19% detectors
    detectors_to_drop = ['SB0236_BHout', 'SB0246_BAout', 'SB121_BBin', 'SB125_BBin', 'SB125_BBout', 'SGN02_BAout', 'SUL62_BA1out', 'SUL62_BDin', 'SUL62_BDout', 'SUL62_BGin', 'SUL62_BHin', 'SUL62_BHout']
    # drop 99% detectors
    detectors_to_drop += ['SB0236_BCout', 'SB1201_BAout']
    # drop near 100% detectors
    detectors_to_drop += ['SB020_BAout', 'SB020_BBin', 'SB020_BCin', 'SB020_BDout']
    det_to_keep = []
    for det in all_detectors:
        if det not in detectors_to_drop:
            det_to_keep.append(det)
    return det_to_keep


def list_df_detectors(data, det_list):
    df_det_list = []
    for det in det_list:
        df_det = get_detector_df(data, det)
        df_det_list.append(df_det)
    return df_det_list


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
    df_detector_null = input_detector[input_detector['count'].isnull()]
    print(df_detector_null.index.unique())

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

    for i in range(num_missing_values):
        df.loc[index + i,'count'] = (df.loc[(index + i) - 2016,'count'] + df.loc[(index + i) - (2016 * 2),'count']) / 2
        df.loc[index + i,'speed'] = (df.loc[(index + i) - 2016,'speed'] + df.loc[(index + i) - (2016 * 2),'speed']) / 2
        df.loc[index + i,'occupancy'] = (df.loc[(index + i) - 2016,'occupancy'] + df.loc[(index + i) - (2016 * 2),'occupancy']) / 2
    return df


def create_speed_variance(input_detector):
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
    input_detector_df = get_detector_df(df, input_detector)
    input_detector_df = roll_up(input_detector_df)
    input_detector_df = gaps_check(input_detector_df)
    input_detector_df = create_speed_variance(input_detector_df)
    return input_detector_df


# ----------------------------------------------------------------------------------------------------
# 4. LSTM PRE-PROCESSING (SEQUENCING DATA)
# ----------------------------------------------------------------------------------------------------

def split_seqs(seqs, n_steps):

    # 0 = 5 min prediction, 1 = 10 min prediction, 2 = 15 min prediction, ...
    added_pred_length = 6

    X, y = list(), list()
    for i in range(len(seqs)):

        end_ix = i + n_steps

        if end_ix + added_pred_length >= len(seqs):
            break

        seq_x, seq_y = seqs[i:end_ix, :], seqs[end_ix + added_pred_length, -1]

        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)


# ----------------------------------------------------------------------------------------------------
# 5. LSTM ARCHITECTURE
# ----------------------------------------------------------------------------------------------------

class MV_LSTM(nn.Module):
    def __init__(self, n_features, seq_length, n_hidden, n_lin_layers, n_lin_nodes):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_length = seq_length
        self.n_hidden = n_hidden  # num hidden states
        self.n_layers = 1  # num LSTM layers
        self.n_linear_layers = n_lin_layers
        self.n_linear_nodes = n_lin_nodes

        self.lstm_layer = nn.LSTM(input_size=self.n_features,
                                  hidden_size=self.n_hidden,
                                  num_layers=self.n_layers,
                                  batch_first=True)

        # LSTM output = (batch_size, seq_length, num_directions * hidden_size)

        if self.n_linear_layers == 1:
            self.linear_layer1 = nn.Linear(self.n_hidden, 1)
        if self.n_linear_layers == 2:
            self.linear_layer1 = nn.Linear(self.n_hidden, self.n_linear_nodes)
            self.linear_layer2 = nn.Linear(self.n_linear_nodes, 1)

    def init_hidden(self, batch_size):
        if CUDA:
            hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).cuda()
            cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).cuda()
        else:
            hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
            cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        lstm_out, self.hidden = self.lstm_layer(x.view(seq_len, batch_size, -1))

        # keep batch size, merge rest, contiguous to solve tensor compatibility error
        # x = lstm_out.contiguous().view(batch_size, -1)

        # lstm_out = F.dropout(lstm_out, training=self.training)

        #return self.linear_layer(lstm_out[-1].view(batch_size, -1))

        if self.n_linear_layers == 1:
            result = self.linear_layer1(lstm_out[-1].view(batch_size, -1))
        if self.n_linear_layers == 2:
            lin1_out = self.linear_layer1(lstm_out[-1].view(batch_size, -1))
            result = self.linear_layer2(lin1_out)

        return result


# ----------------------------------------------------------------------------------------------------
# 6. FUNCTIONS FOR TRAIN / TEST
# ----------------------------------------------------------------------------------------------------

def train(model, X_batches, y_batches, i, criterion, optimizer):
    model.train()

    model.init_hidden(X_batches[i].size(0))
    out = model(X_batches[i])

    train_loss = criterion(out.view(-1), y_batches[i])
    train_loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return train_loss


def test(model, X_batches, y_batches, i, criterion):
    model.eval()

    model.init_hidden(X_batches[i].size(0))
    out = model(X_batches[i])

    return criterion(out.view(-1), y_batches[i])


# ----------------------------------------------------------------------------------------------------
# 7. PLOTTING
# ----------------------------------------------------------------------------------------------------

def plot_results(train_loss_list, test_loss_list):
    plt.title("LSTM MAE loss over 1000 epochs")
    plt.xlabel("Epochs (in 10's)")
    plt.ylabel("MAE Loss")
    plt.plot(train_loss_list, label="train loss")
    plt.plot(test_loss_list, label="test loss")
    plt.legend()
    plt.show()


# ----------------------------------------------------------------------------------------------------
# 8. NORMALISATION
# ----------------------------------------------------------------------------------------------------

def normalize_seq(seq):
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_seq_length = int((len(seq)) * 0.8)

    train_seq = seq[:train_seq_length]
    test_seq = seq[train_seq_length:]

    train_seq = scaler.fit_transform(train_seq.reshape(-1, 1))
    test_seq = scaler.transform(test_seq.reshape(-1, 1))

    sequence = np.concatenate((train_seq, test_seq), axis=0)

    return sequence


# ----------------------------------------------------------------------------------------------------
# 9. INSTANTIATE AND RUN MODEL
# ----------------------------------------------------------------------------------------------------

def run_lstm(p):
    target_detector_df = get_clean_detector(p['target_detector_code'])

    # PNA_203 contains one negative occupancy and count value on index 700, set to zero
    if p['target_detector_code'] == 'PNA_203':
        target_detector_df.loc[700, 'occupancy'] = 0
        target_detector_df.loc[700, 'count'] = 0

    episodes = 1000
    batch_size = 16

    # Make arrays of features
    count_seq = array(target_detector_df['count'])
    speed_seq = array(target_detector_df['speed'])
    speed_var_seq = array(target_detector_df['speed_variance'])
    occupancy_seq = array(target_detector_df['occupancy'])
    weekday_seq = array(target_detector_df['week_day'])
    hour_seq = array(target_detector_df['hour'])

    # Get minimum and maximum occupancy for denormalisation
    min_occupancy, max_occupancy = min(occupancy_seq[:int(len(occupancy_seq)*0.8)]), max(occupancy_seq[:int(len(occupancy_seq)*0.8)])

    # Normalize features
    count_seq = normalize_seq(count_seq)
    speed_seq = normalize_seq(speed_seq)
    speed_var_seq = normalize_seq(speed_var_seq)
    occupancy_seq = normalize_seq(occupancy_seq)
    weekday_seq = normalize_seq(weekday_seq)
    hour_seq = normalize_seq(hour_seq)

    # Reshape to Len x 1
    count_seq = count_seq.reshape((len(count_seq), 1))
    speed_seq = speed_seq.reshape((len(speed_seq), 1))
    speed_var_seq = speed_var_seq.reshape((len(speed_var_seq), 1))
    occupancy_seq = occupancy_seq.reshape((len(occupancy_seq), 1))
    weekday_seq = weekday_seq.reshape((len(weekday_seq), 1))
    hour_seq = hour_seq.reshape((len(hour_seq), 1))

    # Horizontally stack the chosen features to get the actual input samples
    def make_hstack_tuple(count, speed, speed_var, occupancy, weekday, hour):
        list = [count, occupancy]
        if p['include_speed_variance']:
            list += [speed_var]
        if p['include_weekday']:
            list += [weekday]
        if p['include_speed']:
            list += [speed]
        if p['include_hour']:
            list += [hour]
        return tuple(list)
    hstack_input = make_hstack_tuple(count_seq, speed_seq, speed_var_seq, occupancy_seq, weekday_seq, hour_seq)
    hstack_data = hstack(hstack_input)

    # Amount of features is occupancy, count + what is added
    n_features = 2 + p['include_speed_variance'] + p['include_weekday'] + p['include_speed'] + p['include_hour']
    n_timesteps = p['timesteps']

    X, y = split_seqs(hstack_data, n_timesteps)
    # X shape: (col_length, time steps, features)

    # Train test split
    train_set_len = int(len(X) * 0.8)
    X_train = X[:train_set_len]
    X_test = X[train_set_len:]
    y_train = y[:train_set_len]
    y_test = y[train_set_len:]

    X_batches_train, X_batches_test = list(), list()
    y_batches_train, y_batches_test = list(), list()

    for batch_id in range(0, len(X_train), batch_size):
        # Make batches
        input1 = X_train[batch_id:batch_id + batch_size, :, :]
        target = y_train[batch_id:batch_id + batch_size]

        # Turn batches into tensors
        if CUDA:
            X_batch = torch.tensor(input1, dtype=torch.float32).cuda()
            y_batch = torch.tensor(target, dtype=torch.float32).cuda()
        else:
            X_batch = torch.tensor(input1, dtype=torch.float32)
            y_batch = torch.tensor(target, dtype=torch.float32)
        X_batches_train.append(X_batch)
        y_batches_train.append(y_batch)

    for batch_id in range(0, len(X_test), batch_size):
        input1 = X_test[batch_id:batch_id + batch_size, :, :]
        target = y_test[batch_id:batch_id + batch_size]

        if CUDA:
            X_batch = torch.tensor(input1, dtype=torch.float32).cuda()
            y_batch = torch.tensor(target, dtype=torch.float32).cuda()
        else:
            X_batch = torch.tensor(input1, dtype=torch.float32)
            y_batch = torch.tensor(target, dtype=torch.float32)
        X_batches_test.append(X_batch)
        y_batches_test.append(y_batch)

    # Instantiate model
    LSTM = MV_LSTM(n_features, n_timesteps, p['hidden_states'], p['num_linear_layers'], p['num_linear_nodes'])
    if CUDA:
        LSTM.cuda()

    # L1Loss without a given argument is Mean Absolute Error (MAE)
    criterion = nn.L1Loss()

    learning_rate = p['lr']
    optimizer = torch.optim.Adam(LSTM.parameters(), lr=learning_rate)

    # Training
    train_loss_list, test_loss_list = list(), list()
    avg_10ep_loss_list_train, avg_10ep_loss_list_test = list(), list()

    # FOR TESTING
    all_train, all_test = list(), list()

    print("Parameters: ")
    print("Timesteps:", p['timesteps'])
    print("Hidden states:", p['hidden_states'])
    print("Number of linear layers:", p['num_linear_layers'])
    print("Number of linear nodes:", p['num_linear_nodes'])
    print("Speed:", p['include_speed'])
    print("Speed variance:", p['include_speed_variance'])
    print("Weekday:", p['include_weekday'])
    print("Hour:", p['include_hour'])

    print("Max occupancy: ", max_occupancy)
    print("Min occupancy: ", min_occupancy)

    t0 = time.time()
    for episode in range(episodes):

        train_loss = 0
        for i in range(len(X_batches_train)):
            batch_train_loss = train(LSTM, X_batches_train, y_batches_train, i, criterion, optimizer)
            train_loss += batch_train_loss.item()
        train_loss /= (len(X_batches_train))  # Average train loss per epoch

        test_loss = 0
        for i in range(len(X_batches_test)):
            with torch.no_grad():
                batch_test_loss = test(LSTM, X_batches_test, y_batches_test, i, criterion)
                test_loss += batch_test_loss.item()
        test_loss /= (len(X_batches_test))  # Average test loss per epoch

        # Denormalize average losses
        abs_occ_error_train = (train_loss * (max_occupancy - min_occupancy)) + min_occupancy
        abs_occ_error_test = (test_loss * (max_occupancy - min_occupancy)) + min_occupancy

        # lists for plotting
        train_loss_list.append(abs_occ_error_train)
        test_loss_list.append(abs_occ_error_test)

        # FOR TESTING
        all_train.append(abs_occ_error_train)
        all_test.append(abs_occ_error_test)

        t1 = time.time()

        if episode % 10 == 9:
            # Take the average loss of last 10 epochs
            avg_loss_10ep_train = np.average(train_loss_list)
            avg_loss_10ep_test = np.average(test_loss_list)

            print('Episode: ', episode, ', Train loss: ', avg_loss_10ep_train)
            print("Avg denormalized test loss, last 10 epochs: ", avg_loss_10ep_test)
            print(str(int(t1 - t0)) + " seconds for last epoch.")
            print()

            avg_10ep_loss_list_train.append(avg_loss_10ep_train)
            avg_10ep_loss_list_test.append(avg_loss_10ep_test)

            # Reset list
            train_loss_list = []
            test_loss_list = []
        t0 = time.time()

    print("CODE: ", p['target_detector_code'])
    print("MINIMAL TEST LOSS: ", min(avg_10ep_loss_list_test))
    print()

    # PLOT
    plot_results(avg_10ep_loss_list_train, avg_10ep_loss_list_test)

    # FOR TESTING
    #plot_results(all_train, all_test)

    return min(avg_10ep_loss_list_test)


# ----------------------------------------------------------------------------------------------------
# 10. HYPER PARAMETER TUNING
# ----------------------------------------------------------------------------------------------------

# type can be set to fixed / choice
# next keyword has to be value or values respectively
# fixed takes in a single value for value, choice takes in a list for values

best_parameters, best_values, _, _ = optimize(
    parameters=[{'name': 'timesteps', 'type': 'fixed', 'value': 2},
                {'name': 'hidden_states', 'type': 'fixed', 'value': 256},
                {'name': 'num_linear_layers', 'type': 'fixed', 'value': 2},
                {'name': 'num_linear_nodes', 'type': 'fixed', 'value': 64},
                {'name': 'include_speed_variance', 'type': 'fixed', 'value': True},
                {'name': 'include_speed', 'type': 'fixed', 'value': False},
                {'name': 'include_weekday', 'type': 'fixed', 'value': True},
                {'name': 'include_hour', 'type': 'fixed', 'value': False},
                {'name': 'lr', 'type': 'fixed', 'value': 0.001},
                {'name': 'target_detector_code', 'type': 'fixed', 'value': 'ARL_103'}], evaluation_function=run_lstm,
    minimize=True, total_trials=1)
print("best parameters:", best_parameters)
print("best value:", best_values)

# List of all detectors
# ['ARL_103', 'ARL_203', 'BOT_TD2', 'HAL_191', 'HAL_292', 'LOU_110', 'LOU_TD1', 'LOU_TD2', 'MAD_103', 'MAD_203', 'PNA_103', 'PNA_203', 'ROG_TD1', 'ROG_TD2', 'STE_TD1', 'STE_TD2', 'STE_TD3', 'TRO_203', 'TRO_TD1', 'TRO_TD2']

