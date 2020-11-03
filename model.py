from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from __future__ import print_function
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Layer, Input, Lambda
from keras.layers import BatchNormalization, Activation, Concatenate
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from tensorflow.nn import relu
import tensorflow as tf
print(tf.__version__)

def std_norm_along_chs(x) :
    '''Data normalization along the channle axis
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        xn = tensor4d, same shape as x, normalized version of x
    '''
    avg = K.mean(x, axis=-1, keepdims=True)
    std = K.maximum(1e-4, K.std(x, axis=-1, keepdims=True))
    return (x - avg) / std

class SelfCorrelationPercPooling( Layer ) :
    '''Custom Self-Correlation Percentile Pooling Layer
    Arugment:
        nb_pools = int, number of percentile poolings
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        x_pool = tensor4d, (n_samples, n_rows, n_cols, nb_pools)
    '''
    def __init__( self, nb_pools=256, **kwargs ) :
        self.nb_pools = nb_pools
        super( SelfCorrelationPercPooling, self ).__init__( **kwargs )
    def build( self, input_shape ) :
        self.built = True
    def call( self, x, mask=None ) :
        # parse input feature shape
        bsize, nb_rows, nb_cols, nb_feats = K.int_shape( x )
        nb_maps = nb_rows * nb_cols
        # self correlation
        x_3d = K.reshape( x, tf.stack( [ -1, nb_maps, nb_feats ] ) )
        x_corr_3d = tf.matmul( x_3d, x_3d, transpose_a = False, transpose_b = True ) / nb_feats
        x_corr = K.reshape( x_corr_3d, tf.stack( [ -1, nb_rows, nb_cols, nb_maps ] ) )
        # argsort response maps along the translaton dimension
        if ( self.nb_pools is not None ) :
            ranks = K.cast( K.round( tf.linspace( 1., nb_maps - 1, self.nb_pools ) ), 'int32' )
        else :
            ranks = tf.range( 1, nb_maps, dtype = 'int32' )
        x_sort, _ = tf.nn.top_k( x_corr, k = nb_maps, sorted = True )
        # pool out x features at interested ranks
        # NOTE: tf v1.1 only support indexing at the 1st dimension
        x_f1st_sort = K.permute_dimensions( x_sort, ( 3, 0, 1, 2 ) )
        x_f1st_pool = tf.gather( x_f1st_sort, ranks )
        x_pool = K.permute_dimensions( x_f1st_pool, ( 1, 2, 3, 0 ) )
        return x_pool
    def compute_output_shape( self, input_shape ) :
        bsize, nb_rows, nb_cols, nb_feats = input_shape
        nb_pools = self.nb_pools if ( self.nb_pools is not None ) else ( nb_rows * nb_cols - 1 )
        return tuple( [ bsize, nb_rows, nb_cols, nb_pools ] )

class BilinearUpSampling2D( Layer ) :
    '''Custom 2x bilinear upsampling layer
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        x2 = tensor4d, (n_samples, 2*n_rows, 2*n_cols, n_feats)
    '''
    def call( self, x, mask=None ) :
        bsize, nb_rows, nb_cols, nb_filts = K.int_shape(x)
        new_size = tf.constant( [ nb_rows * 2, nb_cols * 2 ], dtype = tf.int32 )
        return tf.image.resize( x, new_size)
    def compute_output_shape( self, input_shape ) :
        bsize, nb_rows, nb_cols, nb_filts = input_shape
        return tuple( [ bsize, nb_rows * 2, nb_cols * 2, nb_filts ] )

class ResizeBack( Layer ) :
    '''Custom bilinear resize layer
    Resize x's spatial dimension to that of r
    
    Input:
        x = tensor4d, (n_samples, n_rowsX, n_colsX, n_featsX )
        r = tensor4d, (n_samples, n_rowsR, n_colsR, n_featsR )
    Output:
        xn = tensor4d, (n_samples, n_rowsR, n_colsR, n_featsX )
    '''
    def call( self, x ) :
        t, r = x
        new_size = [ tf.shape(r)[1], tf.shape(r)[2] ]
        return tf.image.resize( t, new_size)
    def compute_output_shape( self, input_shapes ) :
        tshape, rshape = input_shapes
        return ( tshape[0], ) + rshape[1:3] + ( tshape[-1], )

class Preprocess( Layer ) :
    """Basic preprocess layer for BusterNet
    More precisely, it does the following two things
    1) normalize input image size to (256,256) to speed up processing
    2) substract channel-wise means if necessary
    """
    def call( self, x, mask=None ) :
        # parse input image shape
        bsize, nb_rows, nb_cols, nb_colors = K.int_shape(x)
        if (nb_rows != 256) or (nb_cols !=256) :
            # resize image if different from (256,256)
            x256 = tf.image.resize(x,[256,256],method=tf.image.ResizeMethod.BILINEAR,name='resize')
        else :
            x256 = x
        # substract channel means if necessary
        if K.dtype(x) == 'float32' :
            # input is not a 'uint8' image
            # assume it has already been normalized
            xout = x256
        else :
            # input is a 'uint8' image
            # substract channel-wise means
            xout = preprocess_input( x256 )
        return xout
    def compute_output_shape( self, input_shape ) :
        return (input_shape[0], 256, 256, 3)

# class CAM( Layer ) :
#     def call( self, x, mask=None ) :
#         return x

class CAM( Model ):
  def __init__(self, name=None, **kwargs):
    super(CAM, self).__init__(name=name)

  def call(self, x):
    return x

# You have made a Keras model!
# my_sequential_model = MySequentialModel(name="the_model")

class SAM( Model ) :
  def __init__(self, name=None, **kwargs):
    super(SAM, self).__init__(name=name)

  def call(self, x):
    return x

class ASPP(Model):
  def __init__(self, name=None):
    super(ASPP, self).__init__(name=name)

    self.b1conv1 = Conv2D(230, (3, 3), padding='same',dilation_rate=(4, 4), name='_a1c1l1')
    self.b1norm1 = BatchNormalization(name='_a1c1l2')
    self.b1conv2 = Conv2D(64, (1, 1), padding='same', name='_a1c2l1')
    self.b1norm2 = BatchNormalization(name='_a1c2l2')

    self.b2conv1 = Conv2D(230, (3, 3), padding='same',dilation_rate=(4, 4), name='_a2c1l1')
    self.b2norm1 = BatchNormalization(name='_a2c1l2')
    self.b2conv2 = Conv2D(64, (1, 1), padding='same', name='_a2c2l1')
    self.b2norm2 = BatchNormalization(name='_a2c2l2')

    self.b3conv1 = Conv2D(230, (3, 3), padding='same',dilation_rate=(4, 4), name='_a3c1l1')
    self.b3norm1 = BatchNormalization(name='_a3c1l2')
    self.b3conv2 = Conv2D(64, (1, 1), padding='same', name='_a3c2l1')
    self.b3norm2 = BatchNormalization(name='_a3c2l2')

    self.b4conv1 = Conv2D(230, (3, 3), padding='same',dilation_rate=(4, 4), name='_a4c1l1')
    self.b4norm1 = BatchNormalization(name='_a4c1l2')
    self.b4conv2 = Conv2D(64, (1, 1), padding='same', name='_a4c2l1')
    self.b4norm2 = BatchNormalization(name='_a4c2l2')

  def call(self, input_tensor, training=True):

    #BRANCH 1
    x1 = self.b1conv1(input_tensor)
    x1 = self.b1norm1(x1)
    x1 = relu(x1)

    x1 = self.b1conv2(x1)
    x1 = self.b1norm2(x1)
    x1 = relu(x1)

    #BRANCH 2
    x2 = self.b2conv1(input_tensor)
    x2 = self.b2norm1(x2)
    x2 = relu(x2)

    x2 = self.b2conv2(x2)
    x2 = self.b2norm2(x2)
    x2 = relu(x2)

    #BRANCH 3
    x3 = self.b2conv1(input_tensor)
    x3 = self.b2norm1(x3)
    x3 = relu(x3)

    x3 = self.b2conv2(x3)
    x3 = self.b2norm2(x3)
    x3 = relu(x3)

    #BRANCH 4
    x4 = self.b2conv1(input_tensor)
    x4 = self.b2norm1(x4)
    x4 = relu(x4)

    x4 = self.b2conv2(x4)
    x4 = self.b2norm2(x4)
    x4 = relu(x4)

    x = 0.25*x1 + 0.25*x2 + 0.25*x3 + 0.25*x4
    return x


def create_cmfd_similarity_detection_network( img_shape=(256,256,3),
                                   nb_pools=100,
                                   name='CMSDNet' ) :
    '''Create the similarity branch for copy-move forgery detection
    '''
    #---------------------------------------------------------
    # Input
    #---------------------------------------------------------
    img_input = Input( shape=img_shape, name=name+'_in' )
    #---------------------------------------------------------
    # VGG16 Conv Featex
    #---------------------------------------------------------
    bname = name + '_cnn'
    
    ## FEATURE EXTRACTION

    # Block 1
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name=bname+'_b1c1')(img_input)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name=bname+'_b1c2')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name=bname+'_b1p')(x1)
    # Block 2
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name=bname+'_b2c1')(x1)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name=bname+'_b2c2')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name=bname+'_b2p')(x2)
    # Block 3
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name=bname+'_b3c1')(x2)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name=bname+'_b3c2')(x3)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name=bname+'_b3c3')(x3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name=bname+'_b3p')(x3)
    # Block 4
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(1, 1), name=bname+'_b4c1')(x3)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2), name=bname+'_b4c2')(x4)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=(4, 4), name=bname+'_b4c3')(x4)




    ## Correlation

    x5_1 = CAM(name= bname+'_cam1')(x3)
    x5_1 = SelfCorrelationPercPooling(name=bname+'_corr1')(x5_1)

    x5_2 = CAM(name= bname+'_cam2')(x4)
    x5_2 = SelfCorrelationPercPooling(name=bname+'_corr2')(x5_2)

    # x5_2 = ResizeBack(name=bname+'_restore')([x5_2, x5_1] )

    x6 = Concatenate(axis=-1, name=bname+'_merge')([x5_1, x5_2])

    x6 = BatchNormalization(name=bname+'_bn')(x6)




    ## Mask Decoder Module


    x7 = ASPP(name= bname+'_aspp')(x6)

    # Block 1
    x8 = Conv2D(64, (3, 3),padding='same', name=bname+'_mc1')(x7)
    x8 = BatchNormalization(name=bname+'_bn1')(x8)
    x8 = relu(x8, name=bname+'_relu1')

    x8 = SAM(name= bname+'_sam1')(x8)
    x8  = BilinearUpSampling2D( name=bname+'_bx2')( x8 )

    # Block 2
    x9 = Conv2D(32, (3, 3), padding='same', name=bname+'_mc2')(x8)
    x9 = BatchNormalization(name=bname+'_bn2')(x9)
    x9 = relu(x9, name=bname+'_relu2')

    x9 = SAM(name= bname+'_sam2')(x9)
    x9  = BilinearUpSampling2D( name=bname+'_bx3')( x9 )

    # Block 3
    x10 = Conv2D(16, (3, 3), padding='same', name=bname+'_mc3')(x9)
    x10 = BatchNormalization(name=bname+'_bn3')(x10)
    x10 = relu(x10, name=bname+'_relu3')

    x10 = SAM(name= bname+'_sam3')(x10)
    x10  = BilinearUpSampling2D( name=bname+'_bx4')( x10 )

    # Block 4
    x11 = Conv2D(1, (3, 3), activation='softmax', padding='same', name=bname+'_mc4')(x10)


    #---------------------------------------------------------
    #---------------------------------------------------------
    # End to End
    #---------------------------------------------------------
    model = Model(inputs=img_input, outputs=x11, name=name)
    # print(model.summary())
    return model

def create_CMFD_testing_model( weight_file=None ) :
    '''create a busterNet testing model with pretrained weights
    '''
    # 1. create branch model
    simi_branch = create_cmfd_similarity_detection_network()
    # 2. crop off the last auxiliary task layer
    SimiDet = Model( inputs=simi_branch.inputs,
                     outputs=simi_branch.output,
                     name='CMSD_Net' )
    # 3. define the two-branch BusterNet model
    # 3.a define wrapper inputs
    img_raw = Input( shape=(None,None,3), name='image_in')
    img_in = Preprocess( name='preprocess')( img_raw )
    # 3.b define BusterNet Core
    simi_feat = SimiDet( img_in )
    # mani_feat = ManiDet( img_in )
    # merged_feat = Concatenate(axis=-1, name='merge')([simi_feat, mani_feat])
    # f = merged_feat
    mask_out = Conv2D( 3, (3,3), padding='same', activation='softmax', name='pred_mask')(simi_feat)
    # 3.c define wrapper output
    mask_out = ResizeBack(name='restore')([mask_out, img_raw] )
    # 4. create BusterNet model end-to-end
    model = Model( inputs = img_raw, outputs = mask_out, name = 'Lorem')
    if weight_file is not None :
        try :
            model.load_weights( weight_file )
            print("INFO: successfully load pretrained weights from {}".format( weight_file ) )
        except Exception as e :
            print("INFO: fail to load pretrained weights from {} for reason: {}".format( weight_file, e ))
    return model

print("INFO: this notebook has been tested under keras.version=2.2.2, tensorflow.version=1.8.0")
print("INFO: here is the info your local")
# print("      keras.version={}".format( keras.__version__ ) )
print("      tensorflow.version={}".format( tf.__version__ ) )
print("INFO: consider to the suggested versions if you can't run the following code properly.")