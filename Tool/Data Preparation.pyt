import arcpy
import os
import pandas as pd
import numpy as np


class Toolbox(object):
    def __init__(self):
        self.label = "Mediterranean Network Analysis"
        self.alias = "mednet"
        self.tools = [DataPreparationTool]


class DataPreparationTool(object):
    def __init__(self):
        self.label = "Data Preparation"
        self.description = "Prepares base layers and network edges for analysis"
        self.canRunInBackground = False

    def getParameterInfo(self):
        # Input data - reordered logically
        param0 = arcpy.Parameter(
            displayName="World Boundaries",
            name="world_boundaries",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        )

        param1 = arcpy.Parameter(
            displayName="Cities CSV",
            name="cities_csv",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )
        param1.filter.list = ["csv"]

        param2 = arcpy.Parameter(
            displayName="Diplomatic Edges CSV",
            name="diplomatic_csv",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )
        param2.filter.list = ["csv"]

        param3 = arcpy.Parameter(
            displayName="Trade Edges CSV",
            name="trade_csv",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )
        param3.filter.list = ["csv"]

        # Output parameters - matching the order of inputs
        param4 = arcpy.Parameter(
            displayName="Mediterranean Countries",
            name="med_countries",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )

        param5 = arcpy.Parameter(
            displayName="Cities",
            name="cities",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )

        param6 = arcpy.Parameter(
            displayName="Diplomatic Network",
            name="diplomatic_network",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )

        param7 = arcpy.Parameter(
            displayName="Trade Network",
            name="trade_network",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )

        # New output parameter for combined network
        param8 = arcpy.Parameter(
            displayName="Combined Network",
            name="combined_network",
            datatype="DEFeatureClass",
            parameterType="Derived",
            direction="Output"
        )

        # Add to Map
        param9 = arcpy.Parameter(
            displayName="Add Results To Map",
            name="add_to_map",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        param9.value = True

        return [param0, param1, param2, param3, param4, param5, param6, param7, param8, param9]

    def updateParameters(self, parameters):
        """Set default output paths in project geodatabase"""
        # Get current project folder
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        project_folder = os.path.dirname(aprx.filePath)

        # Create/verify project structure
        gdb_name = "Mediterranean_Network_Analysis.gdb"
        gdb_path = os.path.join(project_folder, gdb_name)

        if not arcpy.Exists(gdb_path):
            try:
                arcpy.management.CreateFileGDB(project_folder, gdb_name)
            except:
                # Just continue if we can't create it now
                pass

        # Set derived output paths
        parameters[4].value = os.path.join(gdb_path, "Mediterranean_Countries")
        parameters[5].value = os.path.join(gdb_path, "Cities")
        parameters[6].value = os.path.join(gdb_path, "Diplomatic_Network")
        parameters[7].value = os.path.join(gdb_path, "Trade_Network")
        parameters[8].value = os.path.join(gdb_path, "Combined_Network")

    def execute(self, parameters, messages):
        try:
            # Get current project folder
            aprx = arcpy.mp.ArcGISProject("CURRENT")
            project_folder = os.path.dirname(aprx.filePath)

            # Get parameter values
            world_boundaries = parameters[0].valueAsText
            cities_csv = parameters[1].valueAsText
            diplomatic_csv = parameters[2].valueAsText
            trade_csv = parameters[3].valueAsText

            med_countries_output = parameters[4].valueAsText
            cities_output = parameters[5].valueAsText
            diplomatic_network = parameters[6].valueAsText
            trade_network = parameters[7].valueAsText
            combined_network = parameters[8].valueAsText

            add_to_map = parameters[9].value

            # Create/verify project geodatabase
            gdb_name = "Mediterranean_Network_Analysis.gdb"
            gdb_path = os.path.join(project_folder, gdb_name)

            if not arcpy.Exists(gdb_path):
                arcpy.management.CreateFileGDB(project_folder, gdb_name)
                arcpy.AddMessage(f"Created geodatabase: {gdb_path}")

            # Set environment
            arcpy.env.workspace = gdb_path
            arcpy.env.overwriteOutput = True

            # 1. Process Mediterranean countries
            arcpy.AddMessage("Creating Mediterranean countries layer...")
            med_countries = [
                "Spain", "Gibraltar", "France", "Monaco", "Italy",
                "Malta", "Slovenia", "Croatia", "Bosnia and Herzegovina",
                "Montenegro", "Albania", "Greece", "Cyprus", "Turkey",
                "Syria", "Lebanon", "Israel", "Palestine", "Egypt",
                "Libyan Arab Jamahiriya", "Tunisia", "Algeria", "Morocco"
            ]

            where_clause = "name IN ('" + "','".join(med_countries) + "')"
            arcpy.analysis.Select(
                world_boundaries,
                med_countries_output,
                where_clause
            )

            # 2. Create cities layer
            arcpy.AddMessage("Creating cities layer...")
            arcpy.management.XYTableToPoint(
                cities_csv,
                cities_output,
                "long",
                "lat",
                coordinate_system=arcpy.SpatialReference(4326)
            )

            # 3. Process network edges
            arcpy.AddMessage("Processing network edges...")

            # Create a dictionary for city coordinates
            cities_df = pd.read_csv(cities_csv)
            diplomatic_df = pd.read_csv(diplomatic_csv)
            trade_df = pd.read_csv(trade_csv)
            city_dict = dict(zip(cities_df['name'], zip(
                cities_df['long'], cities_df['lat'])))

            # Process diplomatic edges
            arcpy.AddMessage("Creating diplomatic network...")

            # Add coordinates to the diplomatic dataframe
            diplomatic_df['from_long'] = diplomatic_df['from'].apply(
                lambda x: city_dict[x][0])
            diplomatic_df['from_lat'] = diplomatic_df['from'].apply(
                lambda x: city_dict[x][1])
            diplomatic_df['to_long'] = diplomatic_df['to'].apply(
                lambda x: city_dict[x][0])
            diplomatic_df['to_lat'] = diplomatic_df['to'].apply(
                lambda x: city_dict[x][1])
            diplomatic_df['from_city'] = diplomatic_df['from']
            diplomatic_df['to_city'] = diplomatic_df['to']
            diplomatic_df['network_type'] = 'Diplomatic'

            # Create temporary diplomatic edges CSV
            temp_dip_csv = os.path.join(
                arcpy.env.scratchFolder, "temp_diplomatic.csv")
            diplomatic_df[['from_city', 'to_city', 'from_long', 'from_lat',
                           'to_long', 'to_lat', 'network_type']].to_csv(temp_dip_csv, index=False)

            # Create diplomatic network lines
            arcpy.XYToLine_management(
                temp_dip_csv,
                diplomatic_network,
                "from_long", "from_lat",
                "to_long", "to_lat",
                line_type="GEODESIC",
                attributes="ATTRIBUTES",
                spatial_reference=arcpy.SpatialReference(4326)
            )

            # Process trade edges
            arcpy.AddMessage("Creating trade network...")

            # Add coordinates to the trade dataframe
            trade_df['from_long'] = trade_df['from'].apply(
                lambda x: city_dict[x][0])
            trade_df['from_lat'] = trade_df['from'].apply(
                lambda x: city_dict[x][1])
            trade_df['to_long'] = trade_df['to'].apply(
                lambda x: city_dict[x][0])
            trade_df['to_lat'] = trade_df['to'].apply(
                lambda x: city_dict[x][1])
            trade_df['from_city'] = trade_df['from']
            trade_df['to_city'] = trade_df['to']
            trade_df['network_type'] = 'Trade'

            # Create temporary trade edges CSV
            temp_trade_csv = os.path.join(
                arcpy.env.scratchFolder, "temp_trade.csv")
            trade_df[['from_city', 'to_city', 'from_long', 'from_lat',
                      'to_long', 'to_lat', 'network_type']].to_csv(temp_trade_csv, index=False)

            # Create trade network lines
            arcpy.XYToLine_management(
                temp_trade_csv,
                trade_network,
                "from_long", "from_lat",
                "to_long", "to_lat",
                line_type="GEODESIC",
                attributes="ATTRIBUTES",
                spatial_reference=arcpy.SpatialReference(4326)
            )

            # Create combined network
            arcpy.AddMessage("Creating combined network...")

            # Merge diplomatic and trade dataframes
            combined_df = pd.concat(
                [diplomatic_df, trade_df], ignore_index=True)

            # Create temporary combined CSV
            temp_combined_csv = os.path.join(
                arcpy.env.scratchFolder, "temp_combined.csv")
            combined_df[['from_city', 'to_city', 'from_long', 'from_lat',
                         'to_long', 'to_lat', 'network_type']].to_csv(temp_combined_csv, index=False)

            # Create combined network lines
            arcpy.XYToLine_management(
                temp_combined_csv,
                combined_network,
                "from_long", "from_lat",
                "to_long", "to_lat",
                line_type="GEODESIC",
                attributes="ATTRIBUTES",
                spatial_reference=arcpy.SpatialReference(4326)
            )

            # 4. Add layers to map if requested
            if add_to_map:
                map_view = aprx.activeMap

                # Add and style Mediterranean countries
                med_layer = map_view.addDataFromPath(med_countries_output)
                med_sym = med_layer.symbology
                # Sand color with transparency
                med_sym.renderer.symbol.color = {'RGB': [238, 214, 175, 100]}
                med_sym.renderer.symbol.outlineColor = {
                    'RGB': [0, 0, 0, 255]}  # Black outline
                med_sym.renderer.symbol.outlineWidth = 0.5
                med_layer.symbology = med_sym

                # Add and style cities with simple symbology
                cities_layer = map_view.addDataFromPath(cities_output)
                city_sym = cities_layer.symbology
                city_sym.renderer.symbol.size = 6
                city_sym.renderer.symbol.color = {
                    'RGB': [128, 0, 0, 255]}  # Maroon color
                city_sym.renderer.symbol.outline = True
                city_sym.renderer.symbol.outlineColor = {
                    'RGB': [255, 255, 255, 255]}  # White outline
                city_sym.renderer.symbol.outlineSize = 0.5
                cities_layer.symbology = city_sym

                # Add and style diplomatic network
                dip_layer = map_view.addDataFromPath(diplomatic_network)
                dip_sym = dip_layer.symbology
                dip_sym.renderer.symbol.color = {'RGB': [0, 51, 153, 120]}
                dip_sym.renderer.symbol.width = 0.5
                dip_layer.symbology = dip_sym

                # Add and style trade network
                trade_layer = map_view.addDataFromPath(trade_network)
                trade_sym = trade_layer.symbology
                trade_sym.renderer.symbol.color = {'RGB': [255, 165, 0, 120]}
                trade_sym.renderer.symbol.width = 0.5
                trade_layer.symbology = trade_sym

                # Add and style combined network
                combined_layer = map_view.addDataFromPath(combined_network)

                # Try to set unique value symbology based on network_type field
                try:
                    combined_sym = combined_layer.symbology

                    if hasattr(combined_sym, 'updateRenderer'):
                        combined_sym.updateRenderer('UniqueValueRenderer')
                        combined_sym.renderer.fields = ["network_type"]

                        # Set diplomatic connections to blue
                        dip_symbol = combined_sym.renderer.getSymbol(
                            "Diplomatic")
                        dip_symbol.color = {'RGB': [0, 51, 153, 120]}
                        dip_symbol.width = 0.5

                        # Set trade connections to orange
                        trade_symbol = combined_sym.renderer.getSymbol("Trade")
                        trade_symbol.color = {'RGB': [255, 165, 0, 120]}
                        trade_symbol.width = 0.5

                        combined_layer.symbology = combined_sym
                    else:
                        # Fallback if unique value renderer not supported
                        comb_sym = combined_layer.symbology
                        comb_sym.renderer.symbol.color = {
                            'RGB': [85, 168, 104, 120]}
                        comb_sym.renderer.symbol.width = 0.5
                        combined_layer.symbology = comb_sym
                except:
                    # Fallback symbology if the unique values approach fails
                    comb_sym = combined_layer.symbology
                    comb_sym.renderer.symbol.color = {
                        'RGB': [85, 168, 104, 120]}
                    comb_sym.renderer.symbol.width = 0.5
                    combined_layer.symbology = comb_sym

                arcpy.AddMessage("Added layers to map")

            arcpy.AddMessage("Data preparation completed!")

        except Exception as e:
            raise
