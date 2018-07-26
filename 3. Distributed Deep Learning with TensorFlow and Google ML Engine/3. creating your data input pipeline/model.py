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

real_feature_columns  = [lat, lng, altitude]
sparse_feature_columns  =  [date_, time_, dt_, lat_buck, lng_buck ]
all_feature_columns = real_feature_columns + sparse_feature_columns

# define input pipeline
def my_input_fn(file_paths, perform_shuffle=True, 
    repeat_count=10000,  batch_size=32):

    def decode_csv(line):
        parsed_line = tf.decode_csv(line, FIELD_DEFAULTS)
        label = tf.convert_to_tensor(parsed_line[-1:])
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Features (but last element)
        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_paths)  # Read text file
                    .skip(1)  # Skip header row
                    .map(decode_csv))  # Transform each elem by decode_csv
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)    
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


pprint(my_input_fn(['data/test/trajectories.csv-00000-of-00104']))