from refactoring.utilities.field.field_creation import Field

import pickle

field_ = Field()
print('HENRYS')
field_.field_shape_file = "../../../Documents/Work/OFPE/Data/Henrys/henrys_bbox.shp"
field_.yld_file = "../../../Documents/Work/OFPE/Data/Henrys/Henrys_yld_2020_1.shp"
field_.as_applied_file = "../../../Documents/Work/OFPE/Data/Henrys/wood_henrys_2018_AA_N.shp"
field_.create_field()
pickle.dump(field_, open('../utilities/saved_fields/Henrys.pickle', 'wb'))
field_1 = Field()
print('35MID')
field_1.field_shape_file = "../../../Documents/Work/OFPE/Data/Sec35Mid/sec35mid_bbox.shp"
field_1.yld_file = "../../../Documents/Work/OFPE/Data/Sec35Mid/Broyles Farm_Broyles Fami_sec 35 middl_Harvest_2020-08-17_00.shp"
field_1.as_applied_file = "../../../Documents/Work/OFPE/Data/Sec35Mid/sec35mid_AA_N_2020_pts.shp"
field_1.create_field()
pickle.dump(field_1, open('../utilities/saved_fields/sec35mid.pickle', 'wb'))
field_2 = Field()
print('35WEST')
field_2.field_shape_file = "../../../Documents/Work/OFPE/Data/Sec35West/sec35west_bbox.shp"
field_2.yld_file = "../../../Documents/Work/OFPE/Data/Sec35West/Broyles Farm_Broyles Fami_sec 35 west_Harvest_2020-08-07_00.shp"
field_2.as_applied_file = "../../../Documents/Work/OFPE/Data/Sec35West/sec35west_AA_N_2020_pts.shp"
field_2.create_field()
pickle.dump(field_2, open('../utilities/saved_fields/sec35west.pickle', 'wb'))