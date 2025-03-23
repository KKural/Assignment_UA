import arcpy
import os


class Toolbox(object):
    def __init__(self):
        self.label = "Join Network Metrics"
        self.alias = "NetworkMetrics"
        self.tools = [JoinNetworkMetricsTool]


class JoinNetworkMetricsTool(object):
    def __init__(self):
        self.label = "Join Network Metrics to Cities"
        self.description = "Joins network metrics to create three city layers"
        self.canRunInBackground = False

    def getParameterInfo(self):
        param0 = arcpy.Parameter(
            displayName="Input Cities Layer",
            name="cities_layer",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        )

        param1 = arcpy.Parameter(
            displayName="Network Metrics Table",
            name="metrics_table",
            datatype="DETable",
            parameterType="Required",
            direction="Input"
        )

        param2 = arcpy.Parameter(
            displayName="Cities Name Field",
            name="cities_field",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        param2.parameterDependencies = [param0.name]

        param3 = arcpy.Parameter(
            displayName="Metrics Name Field",
            name="metrics_field",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        param3.parameterDependencies = [param1.name]

        param4 = arcpy.Parameter(
            displayName="Diplomatic Degree Field",
            name="dip_degree",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        param4.parameterDependencies = [param1.name]

        param5 = arcpy.Parameter(
            displayName="Trade Degree Field",
            name="trade_degree",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        param5.parameterDependencies = [param1.name]

        param6 = arcpy.Parameter(
            displayName="Combined Degree Field",
            name="combined_degree",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        param6.parameterDependencies = [param1.name]

        param7 = arcpy.Parameter(
            displayName="Add Results To Map",
            name="add_to_map",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        param7.value = True

        return [param0, param1, param2, param3, param4, param5, param6, param7]

    def execute(self, parameters, messages):
        try:
            cities_fc = parameters[0].valueAsText
            metrics_table = parameters[1].valueAsText
            cities_field = parameters[2].valueAsText
            metrics_field = parameters[3].valueAsText
            degree_fields = [
                (parameters[4].valueAsText, "Diplomatic_Cities_Degree"),
                (parameters[5].valueAsText, "Trade_Cities_Degree"),
                (parameters[6].valueAsText, "Combined_Cities_Degree")
            ]
            add_to_map = parameters[7].value

            aprx = arcpy.mp.ArcGISProject("CURRENT")
            project_folder = os.path.dirname(aprx.filePath)
            gdb_name = "Mediterranean_Network_Analysis.gdb"
            gdb_path = os.path.join(project_folder, gdb_name)

            if not arcpy.Exists(gdb_path):
                arcpy.management.CreateFileGDB(project_folder, gdb_name)

            arcpy.env.workspace = gdb_path
            arcpy.env.overwriteOutput = True

            created_layers = []

            for _, out_name in degree_fields:
                out_fc = os.path.join(gdb_path, out_name)
                arcpy.management.CopyFeatures(cities_fc, out_fc)
                created_layers.append(out_fc)

            for (degree_field, out_name), out_fc in zip(degree_fields, created_layers):
                try:
                    arcpy.management.JoinField(
                        in_data=out_fc,
                        in_field=cities_field,
                        join_table=metrics_table,
                        join_field=metrics_field,
                        fields=[degree_field]
                    )
                except Exception:
                    continue

            for (degree_field, out_name), out_fc in zip(degree_fields, created_layers):
                top_cities_field = f"Top_10_{degree_field}"
                arcpy.management.AddField(
                    out_fc, top_cities_field, "TEXT", field_length=255)

                city_degrees = []
                with arcpy.da.SearchCursor(out_fc, [cities_field, degree_field]) as cursor:
                    for row in cursor:
                        city_degrees.append((row[0], row[1]))

                top_10_cities = sorted(
                    city_degrees, key=lambda x: x[1], reverse=True)[:10]
                top_10_cities_set = set(city[0] for city in top_10_cities)

                with arcpy.da.UpdateCursor(out_fc, [cities_field, top_cities_field]) as cursor:
                    for row in cursor:
                        if row[0] in top_10_cities_set:
                            row[1] = row[0]
                        else:
                            row[1] = None
                        cursor.updateRow(row)

            if add_to_map:
                try:
                    map_view = aprx.activeMap
                    for (degree_field, out_name), out_fc in zip(degree_fields, created_layers):
                        layer = map_view.addDataFromPath(out_fc)
                        if layer:
                            symbology_layer = os.path.join(
                                project_folder, f"{out_name}.lyrx")
                            if arcpy.Exists(symbology_layer):
                                arcpy.management.ApplySymbologyFromLayer(
                                    layer, symbology_layer)
                except Exception:
                    pass

        except Exception as e:
            arcpy.AddError(str(e))
            raise
