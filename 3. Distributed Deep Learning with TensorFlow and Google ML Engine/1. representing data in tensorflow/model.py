import numpy as np
import tensorflow as tf
from tensorflow import feature_column
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

pprint(all_feature_columns)
