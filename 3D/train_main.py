import os
from utils.threeDCellPol import *

_size = 128 #patch size in the x- and y-directions
_z_size = 64 #patch size in the z-direction
data_dir = '/home/jovyan/discotreino/3DCellPol_Github/dataset/' #directory with the folders "train/images"
                                 #"train/outputs", "val/images", "val/outputs"
save_dir = '/home/jovyan/discotreino/3DCellPol_Github/model/' #directory to save
                                                         #the models and the log file

# Parameters
data_train_configs = {'dim': (_size,_size,_z_size,2),
                                        'mask_dim':(_size,_size,_z_size,4),
                                        'batch_size': 2,
                                        'shuffle': True}

data_val_test_configs = {'dim': (_size,_size,_z_size,2),
                                                'mask_dim':(_size,_size,_z_size,4),
                                                'batch_size': 2,
                                                'shuffle': True}

training_configs = {'initial_learning_rate':0.001,
                'learning_rate_drop':0.8,
                'learning_rate_patience':200,
                'learning_rate_epochs':None, 
                'early_stopping_patience':200,
                'n_epochs':200}

# Generators
train_generator = DataGenerator(data_dir, partition='train', configs=data_train_configs, data_aug=True) 
validation_generator = DataGenerator(data_dir, partition='val', configs=data_train_configs, data_aug=False)

model = threeDCellPol() #training from scratch
#model = load_old_model(model_dir + 'final_3dcellpol.hdf5')) #training from pre-trained model

model.summary()

model = train_model(model=model, model_file=os.path.join(save_dir, 'best_3dcellpol.hdf5'), 
                        logging_file= os.path.join(save_dir, "logs_3dcellpol.log"),
						training_generator=train_generator,
                        validation_generator=validation_generator,
                        steps_per_epoch=train_generator.__len__(),
                        validation_steps=validation_generator.__len__(), **training_configs)

model.save(os.path.join(save_dir,'final_3dcellpol.hdf5'))
