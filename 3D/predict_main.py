from utils.prediction import prediction

model_dir = r'/home/jovyan/discotreino/3DCellPol_Github/model/final_3dcellpol.hdf5' #directory containing the trained model
images_dir = r'/home/jovyan/discotreino/3DCellPol_Github/images' # directory containing the original TEST .tif images
                                                                          # image shape = (x,y,z,channels)
save_dir = r'/home/jovyan/discotreino/3DCellPol_Github/results' #directory where the predicted nucleus-Golgi vectors will be saved

#parameters to test the model
golgi_dist = 2 # optimized using optimization/select_best_thresholds_golgi.py and optimization/roc_curve_golgi.py
nuclei_dist = 5 # optimized using optimization/select_best_thresholds_nuclei.py and optimization/roc_curve_nuclei.py
golgi_prob = 0.7 # optimized using optimization/select_best_thresholds_golgi.py and optimization/roc_curve_golgi.py
nuclei_prob = 0.65 # optimized using optimization/select_best_thresholds_nuclei.py and optimization/roc_curve_nuclei.py
_patch_size = 128 #original patch size
_patch_size_z = 64 #patch size along z
_step_z = 64 #z-step to extract patches with some overlap along the z-direction
_step = 32 #x- and y-step to extract patches with some overlap along the x and y-directions
x_spacing = 0.666 #resolution of the images across x
y_spacing = 0.666 #resolution of the images across y
z_spacing = 0.270 #resolution of the images across z

prediction(model_dir, images_dir, golgi_dist, nuclei_dist, golgi_prob, nuclei_prob, _patch_size, _patch_size_z, _step_z, _step, x_spacing, y_spacing, z_spacing, save_dir)