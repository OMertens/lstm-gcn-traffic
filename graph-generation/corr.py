import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import networkx as nx


def get_detector_df(data, detector):
    df_detector = data[data['detector_code'] == detector]
    df_detector = df_detector.reset_index()
    return df_detector


# Load the pre-processed data frame
df = pd.read_pickle('detector_data.pkl')


def roll_up(input_detector):
    # Roll up to 5 min interval
    roll_up_detector = input_detector.set_index('start_time').resample('5T').mean()
    return roll_up_detector


def gaps_check(input_detector):
    # Check for gaps in data
    df_detector_null = input_detector[input_detector['count'].isnull()]
    #print(df_detector_null.index.unique())

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


def get_detector_list(det_codes):
    input_detectors_list = []
    for code in det_codes:
        clean_det_df = get_clean_detector(code)
        input_detectors_list.append(clean_det_df)
    return input_detectors_list


detector_data_list = get_detector_list(input_detector_codes)
del df  # to save space and ease further calculations

# VANAF HIER : input_detectors_list = lijst van 20 dataframes van elke detector

# Calculate all pairwise correlations between time series of all detectors
def get_corr_matrix(det_list, threshold):
    corr_matrix = np.zeros((len(det_list), len(det_list)))
    for i in range(len(det_list)):
        for j in range(len(det_list)):
            det1 = det_list[i]
            det2 = det_list[j]
            corr_matrix[i, j] = det1.occupancy.corr(det2.occupancy)  # occupancy correlation (Pearson)

    np.fill_diagonal(corr_matrix, np.amin(corr_matrix))  # for correct normalisation: change diagonal from 1 to minimum
    corr_norm = (corr_matrix - np.amin(corr_matrix)) / (np.amax(corr_matrix) - np.amin(corr_matrix))   # normalise
    corr_boolean = corr_norm >= threshold    # keep all pairs to make fully connected graph
    np.fill_diagonal(corr_boolean, False)   # delete self loops
    return corr_boolean, corr_norm


corr_boolean, corr_norm = get_corr_matrix(detector_data_list, 0.8)


def edge_list(boolean, norm):    # make a list of all edges (all pairs of nodes to connect)
    all_edges = []
    edge_weights = []
    for i in range(len(boolean)):
        for j in range(len(boolean)):
            if boolean[i][j]:  # keep cells of matrix with True
                edge = [i, j]
                edge_weights.append(norm[i, j])  # make list with edge weights
                all_edges.append(edge)  # make list with edges
    return all_edges, edge_weights


edges_list, edges_weight_list = edge_list(corr_boolean, corr_norm)
np.savetxt('edges_correlation_not_normalised.csv', edges_list, delimiter=',')  # save to file
np.savetxt('edges_weight_correlation_not_normalised.csv', edges_weight_list, delimiter=',')  # save to file


# Load data from excel
df = pd.read_excel('bxl_detectors.xlsx', usecols="A,D:J,S", nrows=66)
df = df.rename(columns={'Traverse_name': 'ID', 'Description (en)': 'description', 'Orientation': 'orientation',
                        'Number of lanes': 'lanes', 'Lon (WGS 84)': 'lon', 'Lat (WGS 84)': 'lat', 'X (Lb72)': 'x',
                        'Y (Lb72)': 'y', '% Complete': 'complete'})


def detectors_to_keep(data):
    all_detectors = data.ID.unique()
    # drop 0-19% detectors
    detectors_to_drop = ['SB0236_BHout', 'SB0246_BAout', 'SB121_BBin', 'SB125_BBin', 'SB125_BBout', 'SGN02_BAout',
                         'SUL62_BA1out', 'SUL62_BDin', 'SUL62_BDout', 'SUL62_BGin', 'SUL62_BHin', 'SUL62_BHout']
    # drop 99% detectors
    detectors_to_drop += ['SB0236_BCout', 'SB1201_BAout']
    # drop near 100% detectors
    detectors_to_drop += ['SB020_BAout', 'SB020_BBin', 'SB020_BCin', 'SB020_BDout']
    # drop detectors with abnormal behavior
    detectors_to_drop += ['SUL62_BGout', 'TER_TD1']
    # list of 20 used detectors
    detectors_used = ['ARL_103',
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
    det_to_keep = []
    det_to_drop = []
    det_to_use = []
    data['drop'] = 0
    for d in range(len(all_detectors)):
        if all_detectors[d] in detectors_to_drop:
            det_to_drop.append(d),
            data.loc[d, 'drop'] = 'yes'
        elif all_detectors[d] in detectors_used:
            det_to_use.append(d),
            data.loc[d, 'drop'] = 'used'
        else:
            data.loc[d, 'drop'] = 'no'
    for d in range(len(all_detectors)):
        if all_detectors[d] not in detectors_to_drop:
            det_to_keep.append(d),
    return det_to_keep, det_to_use


keep, use = detectors_to_keep(df)
df = df.iloc[use, :]
df.reset_index(drop=True, inplace=True)  # reset index


# Make graph
def coord_graph(data, edges_list):
    G = nx.Graph()
    list_coord = list(zip(data.lon, data.lat))  # make list of tuples (lon, lat)
    dict_coord = {i: list_coord[i] for i in range(0, len(list_coord))}  # make dictionary of node tuples
    tup_edges = [tuple(e) for e in edges_list]  # make list op tuples (node1, node2)
    dict_edge = {i: tup_edges[i] for i in range(0, len(tup_edges))}     # make dictionary of edge tuples

    G.add_nodes_from(dict_coord.keys())     # add nodes to graph
    G.add_edges_from(tup_edges)     # add edges to graph

    plt.figure(1)
    nx.draw(G)  # draw G
    plt.figure(2)
    nx.draw(G, dict_coord)  # draw G with nodes on the right coordinates


coord_graph(df, edges_list)


# Plot graph on map (Folium)
def coord_graph_map(data, edges_list):
    edges_coord = []
    for i in range(len(edges_list)):    # make list of edge tuples with coordinates of both nodes
        edge_lat = [data.lat[edges_list[i][0]], data.lat[edges_list[i][1]]]
        edge_lon = [data.lon[edges_list[i][0]], data.lon[edges_list[i][1]]]
        edge_new = list(zip(edge_lat, edge_lon))
        edges_coord.append(edge_new)

    m = folium.Map(location=[50.845, 4.36], zoom_start=15)
    folium.TileLayer('cartodbpositron').add_to(m)  # background map
    folium.PolyLine(edges_coord).add_to(m)
    for i in range(0, len(data)):
        folium.CircleMarker(data.loc[data.index[i], ['lat', 'lon']],  # coordinates to plot
                        radius=float(2 * df['lanes'][i]),  # size of point based on number of lanes
                        color='green',
                        fill=True,
                        fill_opacity=1,
                        popup=folium.Popup("<b>Traverse ID: " + data['ID'][i] + "</b><br>" + data['description'][i],
                                           min_width=100, max_width=300)).add_to(m)  # popup
    m.save(outfile='map_coord80.html')


coord_graph_map(df, edges_list)



