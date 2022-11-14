import os
import numpy as np
import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))

files = tf.data.Dataset.list_files(os.path.join(dir_path, self.config['training_data_dir'], '*'))
        
dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4)
dataset = dataset.map(map_func=self.parse, num_parallel_calls=self.config['num_parallel_map_calls'])
dataset = dataset.repeat()
dataset = dataset.batch(batch_size=self.config['batch_size'])
dataset = dataset.prefetch(buffer_size=1)
print(dataset)