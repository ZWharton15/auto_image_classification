'''
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.applications.densenet import DenseNet, preprocess_input
'''

import sys
from random import shuffle
from shutil import copyfile
import tqdm
##############################################
#IMPLEMENT HARD/SOFT VOTING ENSEMBLE
##############################################

def get_args():
	image_dir = sys.argv[1]
	output_dir = sys.argv[2]
	should_split = sys.argv[3]
	return image_dir, output_dir, should_split

def split_data(image_dir):
	new_dir = os.path.join(image_dir.replace(os.path.basename(image_dir)),"data")
	
	if not os.isdir(new_dir):
		os.mkdir(new_data)
	if not os.isdir(os.path.join(new_data,"test")):
		os.mkdir(os.path.join(new_data,"test"))
	if not os.isdir(os.path.join(new_data,"train")):
		os.mkdir(os.path.join(new_data,"train"))
	
	for root, dirs, files in os.walk(image_dir):
		if len(dirs) == 0:
			class_data = []
			class_name = os.path.basename(root)
			for fil in files:
				all_data.append(fil)
				pass
			
			all_data = shuffle(all_data)
			test_data = all_data[:len(all_data) * 0.2]
			train_data = all_data[len(all_data) * 0.2:]
			move_images(root, os.path.join(new_dir,"train",class_name), train_data)
			move_images(root, os.path.join(new_dir,"test",class_name), test_data)
			
	pass

def move_images(current_dir, new_dir, files):
	for fil in tqdm(files):
		copyfile(os.path.join(current_dir, fil), os.path.join(new_dir, fil))
	pass


def make_model(model_name, model_output_dir):
	print(model_name)
	model = None
	
	if model_name == "vgg16":
		from train_vgg16 import VGG16_Model
		model = VGG16_Model()
		print("VGG_MODEL FOUND")
	elif model_name == "vgg19":
		from train_vgg19 import VGG19_Model
		model = VGG19_Model()
	elif model_name == "inception_v3":
		from train_inception_v3 import InceptionV3_Model
		model = InceptionV3_Model()
	elif model_name == "inception_resnet_v2":
		from train_inception_resnet_v2 import InceptionResNetV2_Model
		model = InceptionResNetV2_Model()
	elif model_name == "resnet50":
		from train_resnet50 import ResNet50_Model
		model = ResNet50_Model()
	elif model_name == "mobilenet":
		from train_mobilenet import MobileNet_Model
		model = MobileNet_Model()
	elif model_name == "mobilenet_v2":
		from train_mobilenet_v2 import MobileNetV2_Model
		model = MobileNetV2_Model()
	elif model_name == "densenet":
		from train_densenet import DenseNet_Model
		model = DenseNet_Model()
	else:
		raise Exception("{} model architecture not found".format(model_name))
	return model

def get_total_training_images(dir):
	counter = 0
	for root, dirs, files in os.walk(dir):
		counter += len(files)
	return counter

if __name__ == "__main__":
	class_dir, model_outputm should_split	= get_args()
	total_images = get_total_training_images(class_dir)
	
	#Split the users data into a test and training set
	if should_split:
		split_data(class_dir)
	
	#A list of all the models that will/can be used for training
	models_to_train = ['vgg16', 'vgg19', 'inception_v3', 'inception_resnet_v2', 'resnet10', 'mobilenet', 'densenet']
	optimisers = ['SGD', 'Adam', 'RMSprop']
	
	for model_name in models_to_train:
		for current_optimiser in optimisers:
			#import the model object from the correct .py file
			model = make_model(model_name, model_output, )
			#Choose the optimiser that will be used to reduce loss
			model.create_optimiser(optimiser_name=current_optimiser, lr=0.001)
			#train the model
			model.train_generator(5, 16, class_dir, total_images)
			