import pandas as pd
import googlemaps
import numpy as np
import matplotlib.pyplot as plt
import folium
import networkx as nx


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


# Perform request to use the Google Maps API web service
def travel(data, threshold, API_key): #API_key = 'YourAPIKey'  # enter Google Maps API key
    gmaps = googlemaps.Client(key=API_key)

    # Calculate distances
    coord_list = list(zip(data.lat, data.lon))

    # 4 separate requests because API limits to 100 elements per request
    matrix1 = gmaps.distance_matrix(coord_list[0:5], coord_list, mode='driving')
    matrix2 = gmaps.distance_matrix(coord_list[5:10], coord_list, mode='driving')
    matrix3 = gmaps.distance_matrix(coord_list[10:15], coord_list, mode='driving')
    matrix4 = gmaps.distance_matrix(coord_list[15:20], coord_list, mode='driving')

    # Join 4 request to 1 matrix
    def join_API_requests(matrix):
        adj = np.zeros((matrix["origin_addresses"].__len__(), matrix["destination_addresses"].__len__()))
        for i in range(matrix["origin_addresses"].__len__()):
            for j in range(matrix["destination_addresses"].__len__()):
                adj[i, j] = matrix["rows"][i]["elements"][j]["distance"]["value"]
        return adj


    travel = join_API_requests(matrix1)
    travel = np.vstack((travel, join_API_requests(matrix2)))
    travel = np.vstack((travel, join_API_requests(matrix3)))
    travel = np.vstack((travel, join_API_requests(matrix4)))

    #np.savetxt('travel_matrix.csv', travel, delimiter=',')  # save to file to avoid new API requests
    #travel = np.loadtxt('travel_matrix.csv', delimiter=',')  # load the array

    np.fill_diagonal(travel, np.amax(travel))  # for correct normalisation: change diagonal from 0 to maximum
    travel_norm = (travel - np.amin(travel)) / (np.amax(travel) - np.amin(travel))   # normalise
    travel_boolean = travel_norm <= threshold    # keep pairs below threshold
    np.fill_diagonal(travel_boolean, False)   # delete self loops
    return travel_boolean, travel_norm


travel_boolean, travel_norm = travel(df, 0.2, 'YourAPIKey')


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


edges_list, edges_weight_list = edge_list(travel_boolean, travel_norm)
np.savetxt('edges_travel.csv', edges_list, delimiter=',')  # save to file
np.savetxt('edges_weight_travel.csv', edges_weight_list, delimiter=',')  # save to file


# Make graph
def travel_graph(data, edges_list):
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


travel_graph(df, edges_list)


# Plot graph on map (Folium)
def travel_graph_map(data, edges_list):
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
    m.save(outfile='map_travel20.html')


travel_graph_map(df, edges_list)