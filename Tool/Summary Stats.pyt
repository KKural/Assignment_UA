import arcpy
import os
import pandas as pd
import numpy as np


class Toolbox(object):
    def __init__(self):
        self.label = "Summary Statistics Toolbox"
        self.alias = "summarystatistics"
        self.tools = [SummaryStatisticsTool]


class SummaryStatisticsTool(object):
    def __init__(self):
        self.label = "Generate Network Summary Statistics"
        self.description = ("Calculates grouped summary statistics for each network's metrics, "
                            "computes a correlation matrix, and identifies top cities. "
                            "Outputs are saved as CSV files and added as standalone tables to the map. "
                            "Additionally, it pivots the top-cities table into a wide layout.")
        self.canRunInBackground = False

    def getParameterInfo(self):
        # Input Parameter: Cities with Network Metrics feature class
        param0 = arcpy.Parameter(
            displayName="Cities with Network Metrics",
            name="cities_metrics",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Input"
        )

        # Number of top cities (per metric) to report
        param1 = arcpy.Parameter(
            displayName="Number of Top Cities to Report",
            name="top_n",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param1.value = 5  # default: top 5

        # Output Parameters: summary table, top cities table, and correlation matrix table
        param2 = arcpy.Parameter(
            displayName="Output Summary Table",
            name="output_stats",
            datatype="DETable",
            parameterType="Derived",
            direction="Output"
        )

        param3 = arcpy.Parameter(
            displayName="Top Cities Table (Tall)",
            name="top_cities",
            datatype="DETable",
            parameterType="Derived",
            direction="Output"
        )

        param4 = arcpy.Parameter(
            displayName="Correlation Matrix Table",
            name="correlation_table",
            datatype="DETable",
            parameterType="Derived",
            direction="Output"
        )

        # Output folder for CSV files
        param5 = arcpy.Parameter(
            displayName="Output Folder for Summary Report",
            name="output_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        return [param0, param1, param2, param3, param4, param5]

    def updateParameters(self, parameters):
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        project_folder = os.path.dirname(aprx.filePath)
        gdb_path = os.path.join(
            project_folder, "Mediterranean_Network_Analysis.gdb")

        # Set derived outputs
        parameters[2].value = os.path.join(
            gdb_path, "Network_Summary_Statistics")
        parameters[3].value = os.path.join(gdb_path, "Top_Cities")
        parameters[4].value = os.path.join(gdb_path, "Correlation_Matrix")

        # Default output folder is project folder if not altered
        if not parameters[5].altered:
            parameters[5].value = project_folder

    def execute(self, parameters, messages):
        try:
            cities_metrics = parameters[0].valueAsText
            top_n = int(parameters[1].value)
            output_stats = parameters[2].valueAsText
            top_cities = parameters[3].valueAsText
            correlation_table = parameters[4].valueAsText
            output_folder = parameters[5].valueAsText

            # Create output folder if necessary
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            arcpy.env.overwriteOutput = True

            # 1. Load node-level metrics into a DataFrame
            df = self._feature_class_to_df(cities_metrics)
            # (Assumes df has a 'name' field that identifies each city.)

            # 2. Create a grouped summary statistics table
            arcpy.AddMessage("Calculating grouped summary statistics...")
            summary_df = self._calculate_grouped_statistics(df)
            summary_csv = os.path.join(
                output_folder, "Network_Summary_Statistics.csv")
            summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

            # 3. Compute correlation matrix (using all numeric columns)
            arcpy.AddMessage("Calculating correlation matrix...")
            correlation_df = self._calculate_correlation(df)
            correlation_csv = os.path.join(
                output_folder, "Correlation_Matrix.csv")
            correlation_df.to_csv(
                correlation_csv, index=True, encoding="utf-8-sig")

            # 4. Identify top cities based on selected metrics (tall format)
            arcpy.AddMessage("Identifying top cities...")
            top_df = self._identify_top_cities(df, top_n)
            top_csv = os.path.join(output_folder, "Top_Cities.csv")
            top_df.to_csv(top_csv, index=False, encoding="utf-8-sig")

            # 5. Pivot top cities to wide format
            wide_df = self._pivot_top_cities(top_df)
            wide_csv = os.path.join(output_folder, "Top_Cities_Wide.csv")
            wide_df.to_csv(wide_csv, index=False, encoding="utf-8-sig")

            # 6. Create standalone tables in the geodatabase from CSVs
            gdb_path = os.path.dirname(output_stats)  # same GDB as we set
            # summary stats
            arcpy.conversion.TableToTable(
                summary_csv, gdb_path, os.path.basename(output_stats))
            # top cities (tall)
            arcpy.conversion.TableToTable(
                top_csv, gdb_path, os.path.basename(top_cities))
            # correlation matrix
            arcpy.conversion.TableToTable(
                correlation_csv, gdb_path, os.path.basename(correlation_table))

            # top cities (wide) => store as "Top_Cities_Wide"
            wide_table_name = "Top_Cities_Wide"
            arcpy.conversion.TableToTable(wide_csv, gdb_path, wide_table_name)

            # 7. Add the tables to the active map
            aprx = arcpy.mp.ArcGISProject("CURRENT")
            map_view = aprx.activeMap

            # The derived outputs
            map_view.addDataFromPath(output_stats)
            map_view.addDataFromPath(top_cities)
            map_view.addDataFromPath(correlation_table)

            # The new wide table
            wide_table_path = os.path.join(gdb_path, wide_table_name)
            map_view.addDataFromPath(wide_table_path)

        except Exception as e:
            raise

    def _feature_class_to_df(self, fc):
        """Converts an ArcGIS feature class to a pandas DataFrame."""
        field_names = [
            f.name for f in arcpy.ListFields(fc)
            if f.name not in ['Shape', 'OBJECTID', 'Shape_Length', 'Shape_Area']
        ]
        rows = []
        with arcpy.da.SearchCursor(fc, field_names) as cursor:
            for row in cursor:
                rows.append(row)
        return pd.DataFrame(rows, columns=field_names)

    def _calculate_grouped_statistics(self, df):
        """
        Creates a summary table with one row per metric type and, for each network prefix,
        computes min, max, mean, and std. 
        e.g. Diplomatic_In_Degree => belongs to prefix=Diplomatic, metric=In_Degree
        """
        metric_types = [
            "In_Degree", "Out_Degree", "Degree",
            "Betweenness", "Closeness", "Eigenvector",
            "PageRank", "Clustering", "Community"
        ]
        prefixes = ["Diplomatic", "Trade", "Combined"]

        rows = []
        for metric in metric_types:
            row = {"Metric": metric}
            for prefix in prefixes:
                col = f"{prefix}_{metric}"
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors='coerce').dropna()
                    row[f"{prefix}_Min"] = round(
                        vals.min(), 3) if len(vals) else None
                    row[f"{prefix}_Max"] = round(
                        vals.max(), 3) if len(vals) else None
                    row[f"{prefix}_Mean"] = round(
                        vals.mean(), 3) if len(vals) else None
                    row[f"{prefix}_StdDev"] = round(
                        vals.std(), 3) if len(vals) else None
                else:
                    row[f"{prefix}_Min"] = None
                    row[f"{prefix}_Max"] = None
                    row[f"{prefix}_Mean"] = None
                    row[f"{prefix}_StdDev"] = None
            rows.append(row)
        return pd.DataFrame(rows)

    def _calculate_correlation(self, df):
        """Calculates the Pearson correlation matrix for all numeric columns."""
        # Filter numeric columns
        numeric_cols = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        corr_df = df[numeric_cols].corr(method="pearson")
        return corr_df.round(3)

    def _identify_top_cities(self, df, top_n):
        """
        Returns a tall DataFrame with columns:
        [Network, Metric, Rank, City, Value]
        For each (Network,Metric), we sort descending and pick top_n.
        Community metrics are included but treated as categorical.
        """
        # Define all metrics including Community
        candidate_cols = [
            "Diplomatic_In_Degree", "Diplomatic_Out_Degree", "Diplomatic_Degree",
            "Diplomatic_Betweenness", "Diplomatic_Closeness", "Diplomatic_Eigenvector",
            "Diplomatic_PageRank", "Diplomatic_Clustering", "Diplomatic_Community",

            "Trade_In_Degree", "Trade_Out_Degree", "Trade_Degree",
            "Trade_Betweenness", "Trade_Closeness", "Trade_Eigenvector",
            "Trade_PageRank", "Trade_Clustering", "Trade_Community",

            "Combined_In_Degree", "Combined_Out_Degree", "Combined_Degree",
            "Combined_Betweenness", "Combined_Closeness", "Combined_Eigenvector",
            "Combined_PageRank", "Combined_Clustering", "Combined_Community"
        ]

        result_rows = []
        for metric in candidate_cols:
            if metric in df.columns:
                # Sort by metric in descending order
                tmp = df.sort_values(by=metric, ascending=False)
                top_subset = tmp.head(top_n)

                # Parse prefix and metric_type
                parts = metric.split('_', maxsplit=1)
                if len(parts) == 2:
                    prefix, metric_type = parts
                else:
                    prefix = "Unknown"
                    metric_type = metric

                for rank, row in enumerate(top_subset.itertuples(), start=1):
                    city_id = getattr(row, 'name', None)
                    val = getattr(row, metric, 0)
                    # For Community metrics, keep the integer value without rounding
                    if 'Community' in metric:
                        result_rows.append({
                            "Network": prefix,
                            "Metric": metric_type,
                            "Rank": rank,
                            "City": city_id,
                            "Value": int(val)
                        })
                    else:
                        result_rows.append({
                            "Network": prefix,
                            "Metric": metric_type,
                            "Rank": rank,
                            "City": city_id,
                            "Value": round(float(val), 3)
                        })

        final_df = pd.DataFrame(result_rows).sort_values(
            by=['Network', 'Metric', 'Rank']
        )
        return final_df

    def _pivot_top_cities(self, top_df):
        """
        Takes the tall 'top_df' with columns:
            [Network, Metric, Rank, City, Value]
        and pivots to produce columns like:
            [ Metric, Rank,
              Diplomatic_City, Diplomatic_Value,
              Trade_City, Trade_Value,
              Combined_City, Combined_Value ]

        So each row corresponds to one (Metric, Rank).
        """
        # pivot: index=[Metric, Rank], columns=Network, values=[City, Value]
        pivoted = top_df.pivot(
            index=["Metric", "Rank"], columns="Network", values=["City", "Value"])
        # pivoted => multi-level columns:
        #   City          Value
        # N  Diplomatic ...
        # We flatten them
        pivoted.columns = [f"{lvl2}_{lvl1}" for lvl1, lvl2 in pivoted.columns]
        pivoted = pivoted.reset_index()

        # reorder columns
        desired_cols = [
            "Metric", "Rank",
            "City_Diplomatic", "Value_Diplomatic",
            "City_Trade", "Value_Trade",
            "City_Combined", "Value_Combined"
        ]
        existing = [c for c in desired_cols if c in pivoted.columns]
        pivoted = pivoted[existing]

        # rename if desired
        rename_map = {
            "City_Diplomatic": "Diplomatic_City",
            "Value_Diplomatic": "Diplomatic_Value",
            "City_Trade": "Trade_City",
            "Value_Trade": "Trade_Value",
            "City_Combined": "Combined_City",
            "Value_Combined": "Combined_Value"
        }
        pivoted = pivoted.rename(columns=rename_map)

        return pivoted
