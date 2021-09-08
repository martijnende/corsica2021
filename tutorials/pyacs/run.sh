###############################################################################
# TUTORIAL TIME DEPENDENT SLIP INVERSION - MASTER FILE
# J.-M. NOCQUET - GEOAZUR, IRD, UNIV. COTE D AZUR, CNRS, OCA - VALBONNE - FRANCE
# CREATED 2021/09/07
# 
# THIS SCRIPT SUMMURIZES THE STEPS TO PERFORM A KINEMATIC INVERSION OF A SLOW
# SLIP EVENT THAT OCCURRED AT THE ECUADOR SUBDUCTION ZONE IN JUNE/JULY 2021
###############################################################################

# MAKE THE GEOMETRY FROM THE USGS SLAB2.0 SUBDUCTION INTERFACE MODEL
# THIS SCRIPT DISCRETIZES THE SUBDUCTION INTERFACE INTO TRIANGULAR DISLOCATION
# ELEMENTS

cd geometry
pyeq_parametrize_curve_surface_triangles.py -g ../slab2.0/sam_slab2_dep_02.23.18_shifted.grd -n 9 -d 0/35 -e ecuador -b /-83/-78/-2.2/-0.3

# MAKE THE GREEN'S FUNCTION

cd ../green
pyeq_make_green.py -gps_h ./gps_coo.dat -g ../geometry/ecuador_geometry.npy -type tde -method nikkhoo -e green_ecuador

# RUN THE MODEL
cd ../models
pyaks.py -conf pyaks_conf_quick_00.dat

# PRINT RESULTS
pyek_print_result_from_mpck.py -mpck model_quick_00.mpck

# MAKE VARIOUS PLOTS
pyeq_plot_kinematics_shp.py -odir quick_00 -conf conf_plot_custom.dat
