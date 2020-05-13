import folium
from folium.plugins import MarkerCluster
import pandas as pd

# Load data from excel
df = pd.read_excel('bxl_detectors.xlsx', usecols="A,D:H,S", nrows=66)
df = df.rename(columns={'Traverse_name': 'ID', 'Description (en)': 'description', 'Orientation': 'orientation',
                        'Number of lanes': 'lanes', 'Lon (WGS 84)': 'lon', 'Lat (WGS 84)': 'lat',
                        '% Complete': 'complete'})
print(df.head())


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


def generate_map(data):
    color_dict = {'yes': 'red', 'used': 'green', 'no': 'green'}  # color code for plot

    # Function to make map without clusters
    def map_without_cluster(coord):
        m1 = folium.Map(location=[50.84, 4.38], zoom_start=13)
        folium.TileLayer('cartodbpositron').add_to(m1)  # background map
        for i in range(0, len(coord)):
            folium.CircleMarker(coord.loc[coord.index[i], ['lat', 'lon']],  # coordinates to plot
                                radius=float(2 * coord['lanes'][i]),  # size of point based on number of lanes
                                color=color_dict[coord['drop'][i]],  # color circle based on category
                                fill_color=color_dict[coord['drop'][i]],  # color fill based on category
                                fill=True,
                                fill_opacity=1,
                                popup=folium.Popup(
                                    "<b>Traverse ID: " + coord['ID'][i] + "</b><br>" + coord['description'][i],
                                    min_width=100, max_width=300)).add_to(m1)  # popup
        return m1
        #m1.save(outfile='map5_no_clusters_only_used.html')

    # Function to make map with clusters
    def map_with_cluster(coord):
        m2 = folium.Map(location=[50.84, 4.38], zoom_start=13)
        folium.TileLayer('cartodbpositron').add_to(m2)  # background map
        mc = MarkerCluster().add_to(m2)
        for i in range(0, len(coord)):
            folium.CircleMarker(coord.loc[coord.index[i], ['lat', 'lon']],  # coordinates to plot
                                radius=float(2 * coord['lanes'][i]),  # size of point based on number of lanes
                                color=color_dict[coord['drop'][i]],  # color circle based on category
                                fill_color=color_dict[coord['drop'][i]],  # color fill based on category
                                fill=True,
                                fill_opacity=1,
                                popup=folium.Popup(
                                    "<b>Traverse ID: " + coord['ID'][i] + "</b><br>" + coord['description'][i],
                                    min_width=100, max_width=300)).add_to(mc)  # popup
        return m2
        #m2.save(outfile='map5_clusters.html')

    dropped = input("Show all detectors, without dropped ones or only the used ones? All, NoDropped or Used")
    if dropped == "All":
        coord = data.iloc[:, [0, 1, 3, 5, 4, 6, 7]]  # all data and relevant columns

        cluster = input("Do you want clusters? Yes or No")
        if cluster == "No":
            m1 = map_without_cluster(coord)
            m1.save(outfile='map_no_clusters_all_det.html')
        elif cluster == "Yes":
            m2 = map_with_cluster(coord)
            m2.save(outfile='map_clusters_all_det.html')
    elif dropped == "NoDropped":
        coord = data.iloc[keep, [0, 1, 3, 5, 4, 6, 7]]  # data without dropped ones and relevant columns
        coord.reset_index(drop=True, inplace=True)  # reset index

        cluster = input("Do you want clusters? Yes or No")
        if cluster == "No":
            m1 = map_without_cluster(coord)
            m1.save(outfile='map_no_clusters_not_dropped.html')
        elif cluster == "Yes":
            m2 = map_with_cluster(coord)
            m2.save(outfile='map_clusters_not_dropped.html')
    elif dropped == "Used":
        coord = data.iloc[use, [0, 1, 3, 5, 4, 6, 7]]  # data without dropped ones and relevant columns
        coord.reset_index(drop=True, inplace=True)  # reset index

        cluster = input("Do you want clusters? Yes or No")
        if cluster == "No":
            m1 = map_without_cluster(coord)
            m1.save(outfile='map_no_clusters_20used.html')
        elif cluster == "Yes":
            m2 = map_with_cluster(coord)
            m2.save(outfile='map_clusters_20used.html')


generate_map(df)
