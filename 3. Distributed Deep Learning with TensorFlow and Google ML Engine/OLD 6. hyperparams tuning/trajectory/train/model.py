import numpy as np
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.lib.io import file_io
tf.logging.set_verbosity(tf.logging.INFO)
from pprint import pprint 

# DESCRIBE DATASET
# define columns and field defaults
COLUMNS        = ["Lat", "Long", "Altitude","Date_",
                  "Time_", "dt_", "y"]
FIELD_DEFAULTS = [[0.], [0.], [0.], ['na'],
                  ['na'], ['na'], ['na']]
feature_names = COLUMNS[:-1]


def get_features(extras):
        
    # FEATURE COLUMNS
    ## represent feature columns
    # dense feature_columns
    lat      = tf.feature_column.numeric_column("Lat")
    lng      = tf.feature_column.numeric_column("Long")
    altitude = tf.feature_column.numeric_column("Altitude")

    # sparse feature_columns
    date_ = tf.feature_column.categorical_column_with_hash_bucket('Date_', 100)
    time_ = tf.feature_column.categorical_column_with_hash_bucket('Time_', 100)
    dt_ = tf.feature_column.categorical_column_with_hash_bucket('dt_', 100)

    lat_long_buckets = list(np.linspace(-180.0, 180.0, num=30))

    lat_buck  = tf.feature_column.bucketized_column(
        source_column = lat,
        boundaries = lat_long_buckets )

    lng_buck = tf.feature_column.bucketized_column(
        source_column = lng,
        boundaries = lat_long_buckets)

    #  hyperparams: adding features for testing in hyperparm
    crossed_lat_lon = tf.feature_column.crossed_column(
        [lat_buck, lng_buck], 7000)

    lng_buck_embedding = tf.feature_column.embedding_column(
        categorical_column=lng_buck,
        dimension=3)

    lat_buck_embedding = tf.feature_column.embedding_column(
        categorical_column=lat_buck,
        dimension=3)

    crossed_ll_embedding = tf.feature_column.embedding_column(
        categorical_column=crossed_lat_lon,
        dimension=12)

    crossed_all = tf.feature_column.crossed_column(
        ['Lat', 'Long', 'Date_', 'Time_', 'dt_'], 20000)

    crossed_all_embedding = tf.feature_column.embedding_column(
        categorical_column=crossed_all,
        dimension=89)

    date_embedding = tf.feature_column.embedding_column(
        categorical_column=date_,
        dimension=24)

    time_embedding = tf.feature_column.embedding_column(
        categorical_column=time_,
        dimension=16)

    dt_embedding = tf.feature_column.embedding_column(
        categorical_column=dt_,
        dimension=224)

    real_feature_eng = [ lng_buck_embedding, lat_buck_embedding,
            crossed_ll_embedding, date_embedding, time_embedding,
            dt_embedding, crossed_all_embedding]
    sparse_feature_eng = [ lng_buck, lat_buck, crossed_all, crossed_lat_lon]           
    all_feature_eng =  [    lng_buck_embedding, lat_buck_embedding,
                            crossed_ll_embedding, date_embedding, 
                            time_embedding, dt_embedding,
                            crossed_all, crossed_all_embedding]
    all_feature_columns = []
    real_feature_columns  = []
    if (extras == 'NO'):
        real_feature_columns  = [lat, lng, altitude]
        sparse_feature_columns  =  [date_, time_, dt_, lat_buck, lng_buck ]
        all_feature_columns = real_feature_columns + sparse_feature_columns
    else:
        real_feature_columns = [lat, lng, altitude,
           lng_buck_embedding, lat_buck_embedding,
           crossed_ll_embedding, date_embedding, time_embedding,
           dt_embedding, crossed_all_embedding]
        all_feature_columns =  [lat, lng, altitude, date_, time_, dt_, lat_buck,
           lng_buck, lng_buck_embedding, lat_buck_embedding,
           crossed_ll_embedding, date_embedding, time_embedding, dt_embedding,
           crossed_all, crossed_all_embedding]

    return real_feature_columns, all_feature_columns

# define input pipeline
def my_input_fn(file_paths, epochs=10, perform_shuffle=False,  batch_size=32):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, FIELD_DEFAULTS)
        label = tf.convert_to_tensor(parsed_line[-1:])
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything (but last element) are the features
        d = dict(zip(feature_names, features)), label
        return d
    dataset = (tf.data.TextLineDataset(file_paths)  # Read text file
                    .map(decode_csv))  # Transform each elem by decode_csv
    if perform_shuffle:
        dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

# define all class labels
class_labels = ['bike', 'bus', 'car', 
                'driving meet conjestion', 
                'plane', 'subway', 'taxi', 
                'train', 'walk']
                     
def train_eval(traindir, evaldir, batchsize, bucket, epochs, outputdir, hidden_units, feat_eng_cols, job_dir, **kwargs):
    # define classifier config
    classifier_config=tf.estimator.RunConfig(save_checkpoints_steps=10)
    
    # learn_rate = float(learn_rate)
    hidden_units = hidden_units.split(',')
    real_feature_columns, all_feature_columns = get_features(feat_eng_cols)

    optimizer = tf.train.ProximalAdagradOptimizer(
            learning_rate=float(kwargs['learn_rate']),
            l1_regularization_strength=0.1,
            l2_regularization_strength=0.01
            )
    # define classifier
    classifier = tf.estimator.DNNLinearCombinedClassifier(
        linear_feature_columns=all_feature_columns,
        dnn_feature_columns=real_feature_columns,
        dnn_hidden_units = hidden_units,
        n_classes=len(class_labels),
        label_vocabulary=class_labels,
        model_dir=job_dir,
        config=classifier_config, 
        dnn_dropout=float(kwargs['dropout']),
        dnn_optimizer=optimizer
        )
    
   

    # load training and eval files    
    traindata =   [file for file in file_io.get_matching_files(traindir + '/trajectories.csv*')]
    evaldata =    [file for file in file_io.get_matching_files(evaldir + '/trajectories.csv*')]

    # define training and eval params
    train_input = lambda: my_input_fn(
            traindata,
            batch_size=batchsize,
            epochs=epochs,
            perform_shuffle=True
        )

    eval_input = lambda: my_input_fn(
        evaldata,
        batch_size=batchsize,
        perform_shuffle=False,
        epochs=None
    )

    # define training, eval spec for train and evaluate including
    train_spec = tf.estimator.TrainSpec(train_input, max_steps=5000)
    eval_spec = tf.estimator.EvalSpec(eval_input)                                  
    # run training and evaluation
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)