
#CAN REPLACE WITH EARLYSTOPPING CALLBACK
class LossValidation(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
        self.countdown_to_leave = 3
		self.prev_loss = -1

    def on_epoch_end(self, epoch, logs={}):
        if prev_loss == -1:
			self.prev_loss = int(logs.get('loss'))
			break
		else:
			if self.prev_loss - (self.prev_loss * 0.05) < int(logs.get('loss')):
				#loss reduction is minimal, close to potential overfitting
				self.countdown_to_leave -= 1
			elif self.prev_loss < int(logs.get('loss')):
				#loss has increased since last epoch
				self.countdown_to_leave -= 1
			elif self.countdown_to_leave < 2:
				#increase the counter when loss is reduced to prevent shutting down after 3 bad epochs
				self.countdown_to_leave += 1
			
			if self.countdown_to_leave == 0:
				print("Loss not reducing...\nAttempting new optimiser"...)
				self.model.stop_training = True
	

class VGG16_Model():
	
	def __init__(self, img_size=224, is_trainable=True):
		#create the model
		self.img_size = img_size
		self.is_trainable = is_trainable
		
		pass
	
	def create_optimiser(self, optimiser_name, lr=0.001, decay=0.05, momentum=0.95):
		self.lr = lr
		self.optimiser_name = optimiser_name
		
		if self.optimiser_name == "SGD":
			self.optimiser = SGD(lr=lr, decay=decay, momentum=momentum)
		elif self.optimiser_name == "Adam":
			self.optimiser = Adam(lr=lr)
			pass
		elif self.optimiser_name == "RMSprop":
			#Default parameters besides lr are recommend as specified by the Keras documentation
			self.optimiser = RMSprop(lr=lr)
			pass
		else:
			raise Exception("Optimiser name not found")
		pass
	
	def get_model_name(self):
		return "vgg16"
	
	def train_generator(self, epochs, batch_size, data_dir, training_length, checkpoint_freq=1):

		base_model = VGG16(weights='imagenet', input_shape=(img_width, img_height, 3))
		base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
		#    
		# add a global spatial average pooling layer
		x = base_model.output
		predictions = Dense(len(os.listdirs(data_dir)), activation='softmax', name='predictions')(x)
		#    
		#    # this is the model we will train
		model = Model(inputs=base_model.input, outputs=predictions)

		# first: train only the top layers (which were randomly initialized)
		# i.e. freeze all convolutional InceptionV3 layers
		for layer in base_model.layers:
			layer.trainable = self.is_trainable

		# compile the model (should be done *after* setting layers to non-trainable)
		model.compile(optimizer=self.optimiser, loss='categorical_crossentropy', metrics=['accuracy'])
		
		# prepare data augmentation configuration
		train_datagen = image.ImageDataGenerator(
			preprocessing_function=preprocess_input,
			rotation_range=20,
			width_shift_range=0.2,
			height_shift_range=0.2,
			horizontal_flip=True)

		train_generator = train_datagen.flow_from_directory(
			train_data_dir,
			target_size=(self.img_size, self.img_size),
			batch_size=batch_size,
			class_mode='categorical')

		
		csv_logger = CSVLogger('vgg16_train_layers_{}_{}-{}.log'.format(self.is_trainable, self.optimiser_name, self.lr))
		make_checkpoint = ModelCheckpoint(filepath="vgg16_train_layers_{}_{}-{}_weights.{epoch:02d}-{loss:.3f}.hdf5".format(self.is_trainable, self.optimiser_name, self.lr), period=checkpoint_freq)
		loss_val = LossValidation()
		
		# we train our model again (this time fine-tuning the top 2 inception resnet blocks
		# alongside the top Dense layers
		model.fit_generator(
			train_generator,
			steps_per_epoch=(training_length // batch_size),
			epochs=epochs, 
			callbacks=[csv_logger, make_checkpoint, loss_val])

		#save the model
		model.save('vgg16_train_layers_{}_{}-{}.h5'.format(self.is_trainable, self.optimiser_name, self.lr))
