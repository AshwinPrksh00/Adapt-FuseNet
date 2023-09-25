import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from ..utils.convNet import *
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG19 


# Model Builder for Face

def build_face_model(inp_shape, num_classes, mname='resnet50', loss_method='arc'):
  '''
  Building Models for face recognition\n
  Params:\n
  inp_shape: Tuple \n
  Contains the input shape to be passed to the model, preferably three dimensions

  mname: ['vgg', 'mobilenet;, 'resnet50']\n
  Specifies the model backbone to be used in the network\n
  '''
  try:
    face_input = tf.keras.layers.Input(shape = inp_shape[1:], name = 'face_input')
  except:
    print('Using default shape = (48, 128, 128, 1)')
    face_input = tf.keras.layers.Input(shape = (128, 128, 3), name = 'face_input')
  
  label_input = tf.keras.layers.Input(shape=(num_classes,))

  if mname == 'vgg':
    model = VGG19(input_shape=inp_shape[1:], 
                     include_top=False,
                     weights='imagenet',
                     input_tensor=face_input)
    
    model.trainable=True
    
  elif mname == 'resnet50':
    model = ResNet50(input_shape=inp_shape[1:], 
                     include_top=False,
                     weights='imagenet',
                     input_tensor=face_input)
    
    model.trainable=True

  elif mname == 'mobilenet':
    model = MobileNetV2(input_shape=inp_shape[1:], 
                     include_top=False,
                     weights='imagenet',
                     input_tensor=face_input)
    
    model.trainable=True

  else:
    raise Exception('Model not defined. Check the given list of models')
  
  #Freezing some layers of model to prevent over-fitting
  for layer in model.layers[:15]:
    layer.trainable=False

  face_model = model.output
  
  face_model = tf.keras.layers.BatchNormalization(name = 'face_norm_1')(face_model)
  
  face_model = tf.keras.layers.Dropout(0.5, name='face_dropout_1')(face_model)
  
  face_model = tf.keras.layers.Flatten()(face_model)
  
  face_model = tf.keras.layers.Dense(512, kernel_initializer='he_normal',use_bias=False,
                                     kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='face_dense_1')(face_model)
  
  face_model = tf.keras.layers.BatchNormalization(name = 'face_norm_2')(face_model)

  face_model = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(face_model)
  
  loss_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(face_model)
  
  face_model = Model(face_input, loss_layer)
  
  face_model.compile(optimizer = tf.keras.optimizers.Adam(0.001), 
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])
  
  return face_model


# Model Builder for Gait

def build_gait_model(trial, input_shape, config=None, backbone=None, num_classes=124):
    assert len(input_shape)>3, f"Network based on Conv3D layers must contain more than 3 dimensions. Got input shape: {input_shape}"
    gait_input = Input(shape=input_shape[1:]+(1,), name='gait_input')
    if trial is not None:
        # conv3d_units1 = trial.suggest_int("conv3d_units1", 16, 64)
        # conv3dact_units1 = trial.suggest_categorical("conv3dact_units1", ["relu", "selu", "elu", "swish"])
        convlstm_units1 = trial.suggest_int("convlstm_units1", 16, 64)
        convlstm_units2 = trial.suggest_int("convlstm_units2", 16, 32)
        convlstm_units3 = trial.suggest_int("convlstm_units3", 8, 32)
        
        dense_units1 = trial.suggest_int("dense_units1", 600, 1024)
        activation_units1 = trial.suggest_categorical("activation_units1", ["relu", "selu", "elu", "swish"])
        dropout_units1 = trial.suggest_float("dropout_units1", 0.3, 0.7)
        optim = trial.suggest_categorical("optimizer", ['Adam', 'SGD', 'RMSprop', 'AdamW', 'Adadelta', 'Adagrad', 
                                                        'Adamax', 'Adafactor', 'Nadam', 'Ftrl'])
        
    else:
        # print('Non-Optuna Mode Detected...')
        # print('Using Default Parameters for training')
        if config is None:
            # conv3d_units1 = 64
            # conv3dact_units1 = 'relu'
            convlstm_units1 = 64
            convlstm_units2 = 32
            convlstm_units3 = 8
            dense_units1 = 512
            activation_units1 = 'relu'
            dropout_units1 = 0.4986483573
            optim = 'Adam'
        else:
            # conv3d_units1 = config['conv3d_unit']
            # conv3dact_units1 = config['conv3d_act']
            convlstm_units1 = config['convlstm_unit1']
            convlstm_units2 = config['convlstm_unit2']
            convlstm_units3 = config['convlstm_unit3']
            dense_units1 = config['dense_unit1']
            activation_units1 = config['act_unit1']
            dropout_units1 = config['drop_unit1']
            optim = config['optimizer']
     
    #Building Model
    gait_model = ConvLSTM2DBlock(1, convlstm_units1, 3, gait_input)
    gait_model = ConvLSTM2DBlock(2, convlstm_units2, 3, gait_model)
    gait_model = ConvLSTM2DBlock(3, convlstm_units3, 3, gait_model)
    if backbone is not None:
        # out = np.array(gait_model.output)
        gait_model = ConvLSTM2D(3, kernel_size=2, dropout=0.2)(gait_model)
        gait_model = UpSampling2D(3, interpolation='bicubic')(gait_model)
        #gait_reshape = Reshape((-1, 100, gait_model.shape[-1]))(gait_model)
        if backbone == 'vgg':
            bbmodel = VGG19(include_top=False,
                             weights='imagenet',
                             )

            bbmodel.trainable=True

        elif backbone == 'resnet50':
            bbmodel = ResNet50(include_top=False,
                             weights='imagenet'
                              )

            bbmodel.trainable=True

        elif backbone == 'mobilenet':
            bbmodel = MobileNetV2(include_top=False,
                             weights='imagenet')

            bbmodel.trainable=True

        else:
            raise Exception('Model not defined. Check the given list of models')
        
        #Freezing some layers of model to prevent over-fitting
        for layer in bbmodel.layers[:15]:
            layer.trainable=False
        
        gait_model = bbmodel(gait_model)
    
    gait_model = Flatten()(gait_model)
    gait_model = FCBlock(num_classes, dense_units1, dropout_units1, activation_units1, gait_model)
    
    model = Model(gait_input, gait_model)
    
    optims = {
        'Adam': tf.keras.optimizers.Adam(),
        'SGD' : tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.025),
        'RMSprop' : tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0),
        'AdamW' : tf.keras.optimizers.experimental.AdamW(learning_rate=0.001, weight_decay=0.004, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        'Adadelta' : tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07),
        'Adagrad' : tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07),
        'Adamax' : tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        'Adafactor' : tf.keras.optimizers.experimental.Adafactor(learning_rate=0.001, beta_2_decay=-0.8, epsilon_1=1e-30, epsilon_2=0.001, clip_threshold=1.0),
        'Nadam' : tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        'Ftrl' : tf.keras.optimizers.Ftrl(learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0,
                                          l2_regularization_strength=0.0, l2_shrinkage_regularization_strength=0.0, beta=0.0)
    }
    
    #Compile the model
    model.compile(optimizer = optims['Adadelta'],
                 loss='categorical_crossentropy',
                 metrics=['accuracy'] )
    
    if trial is not None:
        print('Trial No:', trial.number)
    else:
        print('Manual Trial')
    #Printing Parameters beforehand
    print(f'Parameters- convlstm_units1: {convlstm_units1}, convlstm_units2: {convlstm_units2}, convlstm_units3: {convlstm_units3}, dense_units1: {dense_units1}, activation_units1: {activation_units1}, dropout_units1: {dropout_units1}, optimizer: {optim}')
    return model

