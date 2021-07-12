"""
Functionality to read and write WKT data.
 TODO: Needs to be made more generalizable.
"""

import ogr, csv, os, re
import osgeo, fiona
from shapely.geometry import Polygon, shape


class ShapeFiles:
    def __init__(self, filename=None):
        self.filename = filename

    def read_shape_file(self, datatype=''):
        datapoints = []
        if datatype == 'yld':
            match_on = re.compile('.*(yl|yield).*')
        elif datatype == 'pro':
            match_on = re.compile('.*pro.*')

        with fiona.open(self.filename) as data:
            column_to_get = ''
            for i, pt in enumerate(data):
                datapoint = [pt['geometry']['coordinates'][0], pt['geometry']['coordinates'][1]]
                if i == 0:
                    for prop in pt['properties']:
                        matched = match_on.match(prop.lower())
                        if matched:
                            print(pt['properties'][prop])
                            if 200 > float(pt['properties'][prop]) >= 0:
                                column_to_get = prop
                                break
                datapoint.append(pt['properties'][column_to_get])
                datapoints.append(datapoint)
        return datapoints



class WKTFiles:
    def __init__(self, filename=None):
        self.filename = filename

    def convert_to_shape(self, coordinatesystem):
        """
        Reads data from CSV file with a WKT column and converts it to a .shp file.
        Below code adapted from:
        https://stackoverflow.com/questions/31927726/converting-a-csv-with-a-wkt-column-to-a-shapefile
        """
        spatialref = osgeo.osr.SpatialReference()  # Spatial reference so the
        spatialref.SetWellKnownGeogCS(coordinatesystem)  # 'EPSG:26912'

        # Create shapefile to write to
        filepath, file_extension = os.path.splitext(self.filename)
        ogr_driver = ogr.GetDriverByName("ESRI Shapefile")
        shapefile = ogr_driver.CreateDataSource(filepath + '.shp')
        wktlayer = shapefile.CreateLayer("layer", spatialref, geom_type=ogr.wkbPolygon)

        # Add the other attribute fields needed:
        field_def = ogr.FieldDefn("ID", ogr.OFTInteger)
        field_def.SetWidth(10)
        wktlayer.CreateField(field_def)

        # Read the feature in your csv file:
        with open(self.filename) as file_input:
            reader = csv.reader(file_input)
            wktindex = 0
            for i, row in enumerate(reader):
                # Find index of wkt column
                if i == 0:
                    for j, cell in enumerate(row):
                        if cell.upper() == "WKT":
                            wktindex = j
                # Transform WKT to ogr geometry to write to shapefile format
                else:
                    polygon = ogr.CreateGeometryFromWkt(str(row[wktindex]))
                    feature = ogr.Feature(wktlayer.GetLayerDefn())
                    feature.SetGeometry(polygon)
                    feature.SetField("ID", i)
                    wktlayer.CreateFeature(feature)

    def grid_read(self):
        """
        Read in csv file containing grid data and transform to a collection of geometry objects.
        Finds WKT column by name: 'geometry'
        """
        cells = []

        filepath, file_extension = os.path.splitext(str(self.filename))
        if file_extension == '.csv':
            with open(self.filename) as file_input:
                reader = csv.reader(file_input)
                wktindex = 0
                for i, row in enumerate(reader):
                    # Find index of wkt column
                    if i == 0:
                        for j, cell in enumerate(row):
                            if cell.lower() == "geometry":
                                wktindex = j
                    # convert wkt polygon to object and add to cells list
                    elif i != 0:
                        ogr_points = []
                        ogr_polygon = ogr.CreateGeometryFromWkt(str(row[wktindex]))
                        ogr_ring = ogr_polygon.GetGeometryRef(0)
                        for i in range(ogr_ring.GetPointCount()):
                            ogr_points.append((ogr_ring.GetPoint(i)[0], ogr_ring.GetPoint(i)[1]))
                        polygon = Polygon(ogr_points)
                        cells.append(polygon)

        else:
            print("file is not a csv")

        return cells

    def create_wkt_file(self, new_filename, _map, field_shape, base_nitrogen=100):
        """
        Creates a csv file with a WKT column to contain the geometry
        and the corresponding yield, protein, and nitrogen information.
        Header line -- 'WKT', 'id', 'yield', 'yield_bin', 'protein', 'protein_bin', 'nitrogen'
        First line -- general field shape geometry with base rate nitrogen.
        Following lines -- Grid cells with assigned values
        """
        with open('./app/static/map_downloads/' + new_filename + '.csv', 'w', newline='') as csvfile:
            fieldnames = ['WKT', 'id', 'yield', 'yield_bin', 'protein', 'protein_bin', 'nitrogen']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            # writing horizontal lines
            id_val = 1
            writer.writerow({'WKT': field_shape, 'id': id_val, 'yield': 0, 'yield_bin': 0,
                             'protein': 0, 'protein_bin': 0, 'nitrogen': base_nitrogen})
            id_val = id_val + 1
            for cell in _map:
                writer.writerow({'WKT': cell.true_bounds, 'id': id_val, 'yield': cell.yield_, 'yield_bin':
                                cell.yield_bin, 'protein': cell.pro_, 'protein_bin': cell.pro_bin,
                                 'nitrogen': cell.nitrogen})
                id_val += 1


if __name__ == '__main__':
    shape_reader = ShapeFiles("C:\\Users\\f24n127\\Documents\\raw-farm-data\\wood-henrys-2018\\WOOD_YL18_YLD_2018_henrys.shp")
    shape_reader.read_shape_file('yld')