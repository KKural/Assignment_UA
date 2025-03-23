import arcpy
import os
import pandas as pd
import numpy as np
import networkx as nx


class Toolbox(object):
    def __init__(self):
        self.label = "Network Analysis Tools"
        self.alias = "networkanalysis"
        self.tools = [NetworkAnalysisTool]


class NetworkAnalysisTool(object):
    def __init__(self):
        self.label = "Calculate Network Metrics"
        self.description = "Calculates network metrics"
        self.canRunInBackground = False

    def getParameterInfo(self):
        # Input Parameters
        param0 = arcpy.Parameter(
            displayName="Diplomatic Network",
            name="diplomatic_network",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        )

        param1 = arcpy.Parameter(
            displayName="Trade Network",
            name="trade_network",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        )

        param2 = arcpy.Parameter(
            displayName="Cities",
            name="cities",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        )

        # Node-level metrics
        param3 = arcpy.Parameter(
            displayName="Node-level Metrics",
            name="node_metrics",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        param3.filter.list = [
            "Degree Centrality", "Betweenness Centrality", "Closeness Centrality",
            "Eigenvector Centrality", "PageRank", "Clustering Coefficient",
            "Community Detection"
        ]
        param3.value = ["Degree Centrality", "Betweenness Centrality"]

        # Network-level metrics
        param4 = arcpy.Parameter(
            displayName="Network-level Metrics",
            name="network_metrics",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        param4.filter.list = [
            "Basic Structure", "Reciprocity", "Component Analysis",
            "Path Metrics", "Clustering", "Community Structure"
        ]
        param4.value = ["Basic Structure", "Component Analysis"]

        # Feature class with node metrics
        param5 = arcpy.Parameter(
            displayName="Cities with Metrics",
            name="cities_metrics",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )

        # Folder for CSVs
        param6 = arcpy.Parameter(
            displayName="Output Folder for CSV Files",
            name="output_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        # Node-level metrics table
        param7 = arcpy.Parameter(
            displayName="Node-level Metrics Table",
            name="node_metrics_table",
            datatype="DETable",
            parameterType="Derived",
            direction="Output"
        )

        # Derived output: Network-level metrics table (in wide format)
        param8 = arcpy.Parameter(
            displayName="Network-Level Metrics Table",
            name="network_metrics_table",
            datatype="DETable",
            parameterType="Derived",
            direction="Output"
        )

        return [param0, param1, param2, param3, param4, param5, param6, param7, param8]

    def updateParameters(self, parameters):
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        project_folder = os.path.dirname(aprx.filePath)
        gdb_path = os.path.join(
            project_folder, "Mediterranean_Network_Analysis.gdb")

        # Node-level output feature class
        parameters[5].value = os.path.join(gdb_path, "Cities_Metrics")

        if not parameters[6].altered:
            parameters[6].value = project_folder

        # Derived outputs for node-level table and network-level table
        parameters[7].value = os.path.join(gdb_path, "Node_Level_Metrics")
        parameters[8].value = os.path.join(gdb_path, "Global_Metrics")
        return

    def execute(self, parameters, messages):
        try:
            diplomatic_network = parameters[0].valueAsText
            trade_network = parameters[1].valueAsText
            cities = parameters[2].valueAsText
            node_metrics_selection = parameters[3].valueAsText.split(";")
            network_metrics_selection = parameters[4].valueAsText.split(";")
            cities_metrics = parameters[5].valueAsText
            output_folder = parameters[6].valueAsText
            node_metrics_table = parameters[7].valueAsText
            network_metrics_table = parameters[8].valueAsText

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            arcpy.env.overwriteOutput = True

            cities_df = self._feature_class_to_df(cities)
            diplomatic_df = self._feature_class_to_df(diplomatic_network)
            trade_df = self._feature_class_to_df(trade_network)

            diplomatic_edges = []
            if 'from_city' in diplomatic_df.columns and 'to_city' in diplomatic_df.columns:
                for _, row in diplomatic_df.iterrows():
                    diplomatic_edges.append((row['from_city'], row['to_city']))

            trade_edges = []
            if 'from_city' in trade_df.columns and 'to_city' in trade_df.columns:
                for _, row in trade_df.iterrows():
                    trade_edges.append((row['from_city'], row['to_city']))

            G_diplomatic = nx.DiGraph()
            G_diplomatic.add_edges_from(diplomatic_edges)

            G_trade = nx.DiGraph()
            G_trade.add_edges_from(trade_edges)

            G_combined = nx.compose(G_diplomatic, G_trade)

            city_metrics = {}
            network_metrics = {
                'Diplomatic Network': {},
                'Trade Network': {},
                'Combined Network': {}
            }

            self._analyze_degree_centrality(
                G_diplomatic, G_trade, G_combined, city_metrics)
            self._analyze_betweenness_centrality(
                G_diplomatic, G_trade, G_combined, city_metrics)
            self._analyze_closeness_centrality(
                G_diplomatic, G_trade, G_combined, city_metrics)
            self._analyze_eigenvector_centrality(
                G_diplomatic, G_trade, G_combined, city_metrics)
            self._analyze_pagerank(G_diplomatic, G_trade,
                                   G_combined, city_metrics)
            self._analyze_clustering_coefficient(
                G_diplomatic, G_trade, G_combined, city_metrics)
            self._detect_communities(
                G_diplomatic, G_trade, G_combined, city_metrics, resolution=1.0, seed=42)
            self._analyze_network_metrics(
                G_diplomatic, G_trade, network_metrics, resolution=1.0, seed=42)

            self._create_cities_metrics_fc(
                cities, city_metrics, cities_metrics)

            city_metrics_df = self._create_city_metrics_df(city_metrics)
            city_metrics_csv = os.path.join(
                output_folder, "Node_Level_Metrics.csv")
            city_metrics_df.to_csv(
                city_metrics_csv, index=False, encoding="utf-8-sig")

            wide_df = self._create_network_metrics_df(network_metrics)
            wide_df = wide_df.round(3)
            global_metrics_csv = os.path.join(
                output_folder, "Global_Metrics.csv")
            wide_df.to_csv(global_metrics_csv, index=False,
                           encoding="utf-8-sig")

            aprx = arcpy.mp.ArcGISProject("CURRENT")
            project_folder = os.path.dirname(aprx.filePath)
            gdb_name = "Mediterranean_Network_Analysis.gdb"
            gdb_path = os.path.join(project_folder, gdb_name)

            node_table_name = "Node_Level_Metrics"
            network_table_name = "Global_Metrics"

            arcpy.conversion.TableToTable(
                city_metrics_csv, gdb_path, node_table_name)
            arcpy.conversion.TableToTable(
                global_metrics_csv, gdb_path, network_table_name)

            map_view = aprx.activeMap
            node_table_path = os.path.join(gdb_path, node_table_name)
            network_table_path = os.path.join(gdb_path, network_table_name)
            map_view.addDataFromPath(node_table_path)
            map_view.addDataFromPath(network_table_path)

            parameters[7].value = node_table_path
            parameters[8].value = network_table_path

        except Exception as e:
            arcpy.AddError(str(e))
            raise

    # --------------------------------------------------------------------------
    # Node-level and Network-level Analysis Functions
    # --------------------------------------------------------------------------
    def _feature_class_to_df(self, fc):
        """Convert an ArcGIS feature class to a pandas DataFrame."""
        field_names = [
            f.name for f in arcpy.ListFields(fc)
            if f.name not in ['Shape', 'OBJECTID', 'Shape_Length', 'Shape_Area']
        ]
        data = []
        with arcpy.da.SearchCursor(fc, field_names) as cursor:
            for row in cursor:
                data.append(row)
        return pd.DataFrame(data, columns=field_names)

    def _analyze_degree_centrality(self, G_diplomatic, G_trade, G_combined, city_metrics):
        dip_in = nx.in_degree_centrality(
            G_diplomatic) if G_diplomatic.nodes() else {}
        dip_out = nx.out_degree_centrality(
            G_diplomatic) if G_diplomatic.nodes() else {}
        tr_in = nx.in_degree_centrality(G_trade) if G_trade.nodes() else {}
        tr_out = nx.out_degree_centrality(G_trade) if G_trade.nodes() else {}
        cb_in = nx.in_degree_centrality(
            G_combined) if G_combined.nodes() else {}
        cb_out = nx.out_degree_centrality(
            G_combined) if G_combined.nodes() else {}

        all_nodes = set(G_diplomatic.nodes()) | set(
            G_trade.nodes()) | set(G_combined.nodes())
        for node in all_nodes:
            if node not in city_metrics:
                city_metrics[node] = {}
            city_metrics[node]['Diplomatic_In_Degree'] = round(
                dip_in.get(node, 0), 3)
            city_metrics[node]['Diplomatic_Out_Degree'] = round(
                dip_out.get(node, 0), 3)
            city_metrics[node]['Diplomatic_Degree'] = round(
                dip_in.get(node, 0) + dip_out.get(node, 0), 3)

            city_metrics[node]['Trade_In_Degree'] = round(
                tr_in.get(node, 0), 3)
            city_metrics[node]['Trade_Out_Degree'] = round(
                tr_out.get(node, 0), 3)
            city_metrics[node]['Trade_Degree'] = round(
                tr_in.get(node, 0) + tr_out.get(node, 0), 3)

            combined_in = cb_in.get(node, 0)
            combined_out = cb_out.get(node, 0)
            city_metrics[node]['Combined_In_Degree'] = round(combined_in, 3)
            city_metrics[node]['Combined_Out_Degree'] = round(combined_out, 3)
            city_metrics[node]['Combined_Degree'] = round(
                combined_in + combined_out, 3)

    def _analyze_betweenness_centrality(self, G_diplomatic, G_trade, G_combined, city_metrics):
        N_dip = len(G_diplomatic)
        N_trade = len(G_trade)
        N_combined = len(G_combined)
        dip_bet = nx.betweenness_centrality(
            G_diplomatic, normalized=False) if N_dip > 2 else {}
        tr_bet = nx.betweenness_centrality(
            G_trade, normalized=False) if N_trade > 2 else {}
        cb_bet = nx.betweenness_centrality(
            G_combined, normalized=False) if N_combined > 2 else {}

        max_possible_betweenness_dip = (
            N_dip - 1) * (N_dip - 2) if N_dip > 2 else 1
        max_possible_betweenness_trade = (
            N_trade - 1) * (N_trade - 2) if N_trade > 2 else 1
        max_possible_betweenness_combined = (
            N_combined - 1) * (N_combined - 2) if N_combined > 2 else 1

        all_nodes = set(G_diplomatic.nodes()) | set(
            G_trade.nodes()) | set(G_combined.nodes())
        for node in all_nodes:
            if node not in city_metrics:
                city_metrics[node] = {}
            city_metrics[node]['Diplomatic_Betweenness'] = round(
                dip_bet.get(node, 0) / max_possible_betweenness_dip, 3)
            city_metrics[node]['Trade_Betweenness'] = round(
                tr_bet.get(node, 0) / max_possible_betweenness_trade, 3)
            city_metrics[node]['Combined_Betweenness'] = round(
                cb_bet.get(node, 0) / max_possible_betweenness_combined, 3)

    def _analyze_closeness_centrality(self, G_diplomatic, G_trade, G_combined, city_metrics):
        dip_close = nx.closeness_centrality(
            G_diplomatic, wf_improved=True) if G_diplomatic.nodes() else {}
        tr_close = nx.closeness_centrality(
            G_trade, wf_improved=True) if G_trade.nodes() else {}
        cb_close = nx.closeness_centrality(
            G_combined, wf_improved=True) if G_combined.nodes() else {}

        all_nodes = set(G_diplomatic.nodes()) | set(
            G_trade.nodes()) | set(G_combined.nodes())
        for node in all_nodes:
            if node not in city_metrics:
                city_metrics[node] = {}
            city_metrics[node]['Diplomatic_Closeness'] = round(
                dip_close.get(node, 0), 3)
            city_metrics[node]['Trade_Closeness'] = round(
                tr_close.get(node, 0), 3)
            city_metrics[node]['Combined_Closeness'] = round(
                cb_close.get(node, 0), 3)

    def _analyze_eigenvector_centrality(self, G_diplomatic, G_trade, G_combined, city_metrics):
        try:
            dip_eig = nx.eigenvector_centrality(
                G_diplomatic, max_iter=1000) if G_diplomatic.nodes() else {}
        except:
            dip_eig = {}

        try:
            tr_eig = nx.eigenvector_centrality(
                G_trade, max_iter=1000) if G_trade.nodes() else {}
        except:
            tr_eig = {}

        try:
            cb_eig = nx.eigenvector_centrality(
                G_combined, max_iter=1000) if G_combined.nodes() else {}
        except:
            cb_eig = {}

        all_nodes = set(G_diplomatic.nodes()) | set(
            G_trade.nodes()) | set(G_combined.nodes())
        for node in all_nodes:
            if node not in city_metrics:
                city_metrics[node] = {}
            city_metrics[node]['Diplomatic_Eigenvector'] = round(
                dip_eig.get(node, 0), 3)
            city_metrics[node]['Trade_Eigenvector'] = round(
                tr_eig.get(node, 0), 3)
            city_metrics[node]['Combined_Eigenvector'] = round(
                cb_eig.get(node, 0), 3)

    def _analyze_pagerank(self, G_diplomatic, G_trade, G_combined, city_metrics):
        dip_rank = nx.pagerank(G_diplomatic) if G_diplomatic.nodes() else {}
        tr_rank = nx.pagerank(G_trade) if G_trade.nodes() else {}
        cb_rank = nx.pagerank(G_combined) if G_combined.nodes() else {}

        all_nodes = set(G_diplomatic.nodes()) | set(
            G_trade.nodes()) | set(G_combined.nodes())
        for node in all_nodes:
            if node not in city_metrics:
                city_metrics[node] = {}
            city_metrics[node]['Diplomatic_PageRank'] = round(
                dip_rank.get(node, 0), 3)
            city_metrics[node]['Trade_PageRank'] = round(
                tr_rank.get(node, 0), 3)
            city_metrics[node]['Combined_PageRank'] = round(
                cb_rank.get(node, 0), 3)

    def _analyze_clustering_coefficient(self, G_diplomatic, G_trade, G_combined, city_metrics):
        dip_und = G_diplomatic.to_undirected()
        tr_und = G_trade.to_undirected()
        cb_und = G_combined.to_undirected()
        dip_clust = nx.clustering(dip_und) if dip_und.nodes() else {}
        tr_clust = nx.clustering(tr_und) if tr_und.nodes() else {}
        cb_clust = nx.clustering(cb_und) if cb_und.nodes() else {}

        all_nodes = set(G_diplomatic.nodes()) | set(
            G_trade.nodes()) | set(G_combined.nodes())
        for node in all_nodes:
            if node not in city_metrics:
                city_metrics[node] = {}
            city_metrics[node]['Diplomatic_Clustering'] = round(
                dip_clust.get(node, 0), 3)
            city_metrics[node]['Trade_Clustering'] = round(
                tr_clust.get(node, 0), 3)
            city_metrics[node]['Combined_Clustering'] = round(
                cb_clust.get(node, 0), 3)

    def _detect_communities(self, G_diplomatic, G_trade, G_combined, city_metrics, resolution=1.0, seed=42):
        """Detect communities with Louvain, specifying resolution & seed for consistent results."""
        dip_und = G_diplomatic.to_undirected()
        tr_und = G_trade.to_undirected()
        cb_und = G_combined.to_undirected()

        # Diplomatic
        if dip_und.nodes():
            try:
                dip_coms = list(nx.community.louvain_communities(
                    dip_und, resolution=resolution, seed=seed
                ))
                for i, community in enumerate(dip_coms):
                    for node in community:
                        if node not in city_metrics:
                            city_metrics[node] = {}
                        city_metrics[node]['Diplomatic_Community'] = i + 1
            except Exception as e:
                for node in dip_und.nodes():
                    if node not in city_metrics:
                        city_metrics[node] = {}
                    city_metrics[node]['Diplomatic_Community'] = 0

        # Trade
        if tr_und.nodes():
            try:
                tr_coms = list(nx.community.louvain_communities(
                    tr_und, resolution=resolution, seed=seed
                ))
                for i, community in enumerate(tr_coms):
                    for node in community:
                        if node not in city_metrics:
                            city_metrics[node] = {}
                        city_metrics[node]['Trade_Community'] = i + 1
            except Exception as e:
                for node in tr_und.nodes():
                    if node not in city_metrics:
                        city_metrics[node] = {}
                    city_metrics[node]['Trade_Community'] = 0

        # Combined
        if cb_und.nodes():
            try:
                cb_coms = list(nx.community.louvain_communities(
                    cb_und, resolution=resolution, seed=seed
                ))
                for i, community in enumerate(cb_coms):
                    for node in community:
                        if node not in city_metrics:
                            city_metrics[node] = {}
                        city_metrics[node]['Combined_Community'] = i + 1
            except Exception as e:
                for node in cb_und.nodes():
                    if node not in city_metrics:
                        city_metrics[node] = {}
                    city_metrics[node]['Combined_Community'] = 0

    def _analyze_network_metrics(self, G_diplomatic, G_trade, network_metrics, resolution=1.0, seed=42):
        """Analyze basic network metrics plus Louvain community count with a fixed resolution & seed."""
        G_combined = nx.compose(G_diplomatic, G_trade)

        for name, G in [
            ('Diplomatic Network', G_diplomatic),
            ('Trade Network', G_trade),
            ('Combined Network', G_combined)
        ]:
            if not G.nodes():
                network_metrics[name] = {
                    'Nodes': 0,
                    'Edges': 0,
                    'Average Degree': 0,
                    'Graph Density': 0,
                    'Strongly Connected Components': 0,
                    'Largest Component Size': 0,
                    'Average Path Length': 0,
                    'Network Diameter': 0,
                    'Communities': 0,
                    'Modularity': 0
                }
                continue

            network_metrics[name] = {
                'Nodes': G.number_of_nodes(),
                'Edges': G.number_of_edges(),
            }
            # Average degree
            avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes()
            network_metrics[name]['Average Degree'] = round(avg_deg, 3)

            # Graph density
            network_metrics[name]['Graph Density'] = round(nx.density(G), 3)

            # Reciprocity for Diplomatic/Trade only
            if name != 'Combined Network':
                network_metrics[name]['Reciprocity'] = round(
                    nx.reciprocity(G), 3)

            # SCC
            scc_count = nx.number_strongly_connected_components(G)
            network_metrics[name]['Strongly Connected Components'] = scc_count

            # Largest component size
            try:
                largest_cc = max(nx.strongly_connected_components(G), key=len)
                network_metrics[name]['Largest Component Size'] = len(
                    largest_cc)

                if len(largest_cc) > 1:
                    largest_subgraph = G.subgraph(largest_cc)
                    apl = nx.average_shortest_path_length(largest_subgraph)
                    network_metrics[name]['Average Path Length'] = round(
                        apl, 3)
                    network_metrics[name]['Network Diameter'] = nx.diameter(
                        largest_subgraph)
                else:
                    network_metrics[name]['Average Path Length'] = 0
                    network_metrics[name]['Network Diameter'] = 0
            except ValueError:
                network_metrics[name]['Largest Component Size'] = 0
                network_metrics[name]['Average Path Length'] = 0
                network_metrics[name]['Network Diameter'] = 0

            # Clustering metrics (undirected)
            undirected_G = G.to_undirected()
            network_metrics[name]['Average Clustering'] = round(
                nx.average_clustering(undirected_G), 3)
            network_metrics[name]['Transitivity'] = round(
                nx.transitivity(undirected_G), 3)

            # Community detection
            try:
                comms = list(nx.community.louvain_communities(
                    undirected_G, resolution=resolution, seed=seed
                ))
                network_metrics[name]['Communities'] = len(comms)
                mod_val = nx.community.modularity(undirected_G, comms)
                network_metrics[name]['Modularity'] = round(mod_val, 3)
            except:
                network_metrics[name]['Communities'] = 0
                network_metrics[name]['Modularity'] = 0

    def _create_cities_metrics_fc(self, cities_fc, city_metrics, output_fc):
        arcpy.conversion.FeatureClassToFeatureClass(
            cities_fc, os.path.dirname(output_fc), os.path.basename(output_fc))
        metric_fields = [
            'Diplomatic_In_Degree', 'Diplomatic_Out_Degree', 'Diplomatic_Degree',
            'Diplomatic_Betweenness', 'Diplomatic_Closeness', 'Diplomatic_Eigenvector',
            'Diplomatic_PageRank', 'Diplomatic_Clustering', 'Diplomatic_Community',

            'Trade_In_Degree', 'Trade_Out_Degree', 'Trade_Degree',
            'Trade_Betweenness', 'Trade_Closeness', 'Trade_Eigenvector',
            'Trade_PageRank', 'Trade_Clustering', 'Trade_Community',

            'Combined_In_Degree', 'Combined_Out_Degree', 'Combined_Degree',
            'Combined_Betweenness', 'Combined_Closeness', 'Combined_Eigenvector',
            'Combined_PageRank', 'Combined_Clustering', 'Combined_Community'
        ]
        for field in metric_fields:
            if 'Community' in field:
                arcpy.management.AddField(output_fc, field, "LONG")
            else:
                arcpy.management.AddField(output_fc, field, "DOUBLE")

        with arcpy.da.UpdateCursor(output_fc, ['name'] + metric_fields) as cursor:
            for row in cursor:
                city_name = row[0]
                if city_name in city_metrics:
                    for i, field in enumerate(metric_fields):
                        default_val = 0 if 'Community' in field else 0.0
                        row[i +
                            1] = city_metrics[city_name].get(field, default_val)
                    cursor.updateRow(row)

    def _create_city_metrics_df(self, city_metrics):
        rows = []
        for city, metrics in city_metrics.items():
            row = {'City': city}
            row.update(metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def _create_network_metrics_df(self, network_metrics):
        global_metrics = [
            'Nodes',
            'Edges',
            'Average Degree',
            'Graph Density',
            'Strongly Connected Components',
            'Largest Component Size',
            'Network Diameter',
            'Communities',
            'Modularity'
        ]

        rows = []
        for net_name in ["Diplomatic Network", "Trade Network", "Combined Network"]:
            data_dict = network_metrics.get(net_name, {})
            for metric in global_metrics:
                if metric in data_dict:
                    rows.append({
                        'Network': net_name,
                        'Metric': metric,
                        'Value': data_dict[metric]
                    })

        long_df = pd.DataFrame(rows)
        wide_df = long_df.pivot(
            index="Metric", columns="Network", values="Value")
        wide_df = wide_df.reindex(global_metrics)
        wide_df = wide_df[["Diplomatic Network",
                           "Trade Network", "Combined Network"]]
        wide_df.columns.name = None
        wide_df = wide_df.reset_index()
        return wide_df
