import torch
import tensorflow_datasets as tfds
import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUTO = tf.data.experimental.AUTOTUNE

BAND_STATS = {
            'S2':{
                'mean': {
                    'B01': 340.76769064,
                    'B02': 429.9430203,
                    'B03': 614.21682446,
                    'B04': 590.23569706,
                    'B05': 950.68368468,
                    'B06': 1792.46290469,
                    'B07': 2075.46795189,
                    'B08': 2218.94553375,
                    'B8A': 2266.46036911,
                    'B09': 2246.0605464,
                    'B11': 1594.42694882,
                    'B12': 1009.32729131
                },
                'std': {
                    'B01': 554.81258967,
                    'B02': 572.41639287,
                    'B03': 582.87945694,
                    'B04': 675.88746967,
                    'B05': 729.89827633,
                    'B06': 1096.01480586,
                    'B07': 1273.45393088,
                    'B08': 1365.45589904,
                    'B8A': 1356.13789355,
                    'B09': 1302.3292881,
                    'B11': 1079.19066363,
                    'B12': 818.86747235
                }
            }
        }

def feed_dict(parsed_example):
        B01  = (tf.cast(parsed_example['B01'], dtype=tf.float32) - BAND_STATS['S2']['mean']['B01']) / BAND_STATS['S2']['std']['B01']        
        B02  = (tf.cast(parsed_example['B02'], dtype=tf.float32) - BAND_STATS['S2']['mean']['B02']) / BAND_STATS['S2']['std']['B02']
        B03  = (tf.cast(parsed_example['B03'], dtype=tf.float32) - BAND_STATS['S2']['mean']['B03']) / BAND_STATS['S2']['std']['B03']
        B04  = (tf.cast(parsed_example['B04'], dtype=tf.float32) - BAND_STATS['S2']['mean']['B04']) / BAND_STATS['S2']['std']['B04']
        B05  = (tf.cast(parsed_example['B05'], dtype=tf.float32) - BAND_STATS['S2']['mean']['B05']) / BAND_STATS['S2']['std']['B05']
        B06  = (tf.cast(parsed_example['B06'], dtype=tf.float32) - BAND_STATS['S2']['mean']['B06']) / BAND_STATS['S2']['std']['B06']
        B07  = (tf.cast(parsed_example['B07'], dtype=tf.float32) - BAND_STATS['S2']['mean']['B07']) / BAND_STATS['S2']['std']['B07']
        B08  = (tf.cast(parsed_example['B08'], dtype=tf.float32) - BAND_STATS['S2']['mean']['B08']) / BAND_STATS['S2']['std']['B08']
        B8A  = (tf.cast(parsed_example['B8A'], dtype=tf.float32) - BAND_STATS['S2']['mean']['B8A']) / BAND_STATS['S2']['std']['B8A']
        B09  = (tf.cast(parsed_example['B09'], dtype=tf.float32) - BAND_STATS['S2']['mean']['B09']) / BAND_STATS['S2']['std']['B09']
        B11  = (tf.cast(parsed_example['B11'], dtype=tf.float32) - BAND_STATS['S2']['mean']['B11']) / BAND_STATS['S2']['std']['B11']
        B12  = (tf.cast(parsed_example['B12'], dtype=tf.float32) - BAND_STATS['S2']['mean']['B12']) / BAND_STATS['S2']['std']['B12']

        return {
                'B01': B01,
                'B02': B02,
                'B03': B03,
                'B04': B04,
                'B05': B05,
                'B06': B06,
                'B07': B07,
                'B08': B08,
                'B8A': B8A,
                'B09': B09,
                'B11': B11,
                'B12': B12
            }

def parse_tfrecord(example_proto, label_type, RGB=False):
    nb_class = 43 if label_type == 'original' else 19

    feature_description = {
        'B01': tf.io.FixedLenFeature([20*20], tf.int64),
        'B02': tf.io.FixedLenFeature([120*120], tf.int64),
        'B03': tf.io.FixedLenFeature([120*120], tf.int64),
        'B04': tf.io.FixedLenFeature([120*120], tf.int64),
        'B05': tf.io.FixedLenFeature([60*60], tf.int64),
        'B06': tf.io.FixedLenFeature([60*60], tf.int64),
        'B07': tf.io.FixedLenFeature([60*60], tf.int64),
        'B08': tf.io.FixedLenFeature([120*120], tf.int64),
        'B8A': tf.io.FixedLenFeature([60*60], tf.int64),
        'B09': tf.io.FixedLenFeature([20*20], tf.int64),
        'B11': tf.io.FixedLenFeature([60*60], tf.int64),
        'B12': tf.io.FixedLenFeature([60*60], tf.int64),
        # 'patch_name_s2': tf.io.VarLenFeature(dtype=tf.string),
        # label_type + '_labels': tf.io.VarLenFeature(dtype=tf.string),
        'original_labels_multi_hot': tf.io.FixedLenFeature([nb_class], tf.int64)
    }

    bands_20 = ['B01', 'B09']
    bands_60 = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    bands_120 = ['B02', 'B03', 'B04', 'B08']

    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    labels = parsed_example.pop('original_labels_multi_hot')
    parsed_example = feed_dict(parsed_example)
    # parsed_example[label_type + '_labels'] = tf.sparse.to_dense(parsed_example[label_type + '_labels'])

    bands = feature_description.keys() - ['original_labels_multi_hot']

    band_tensors = []
    if RGB:
        band_tensors = [parsed_example['B04'], parsed_example['B03'], parsed_example['B02']]
    else:
        for band in bands:
            if band in bands_20:
                # interpolate 20x20 vector to 120x120
                parsed_example[band] = tf.reshape(parsed_example[band], [20, 20, -1])
                # print(parsed_example[band].shape)
                parsed_example[band] = tf.image.resize(parsed_example[band], [120, 120])
            elif band in bands_60:
                # interpolate 60x60 vector to 120x120
                parsed_example[band] = tf.reshape(parsed_example[band], [60, 60, -1])
                # print(parsed_example[band].shape)
                parsed_example[band] = tf.image.resize(parsed_example[band], [120, 120])
            band_tensors.append(tf.reshape(parsed_example[band], [-1]))
            
    concatenated_bands = tf.concat(band_tensors, axis=0)

    num_bands = len(band_tensors)
    height, width = 120, 120  # Adjust based on your specific use case
    concatenated_bands = tf.reshape(concatenated_bands, [num_bands, height, width])

    return concatenated_bands, labels


def get_dataset(file, batch_size=16, repeat=False, cache=False, shuffle=False, shuffle_size=1000, label_type="original", return_image_ids=True, RGB=False):
    ds = tf.data.TFRecordDataset(file)
    if cache:
        # You'll need around 15GB RAM if you'd like to cache val dataset, and 50~60GB RAM for train dataset.
        ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(shuffle_size)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    ds = ds.map(lambda x: parse_tfrecord(example_proto=x, label_type=label_type, RGB=RGB), num_parallel_calls=AUTO)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTO)
    return tfds.as_numpy(ds)

def counts_tfrecord(file):
    return sum(1 for _ in tf.data.TFRecordDataset(file))

class BigEarthNetDataLoader:
    def __init__(self, file, batch_size=32, cache=False, repeat=False, shuffle=False, label_type="original", return_image_ids=True, RGB=False, seed=42):
        tf.keras.utils.set_random_seed(seed)
        self.ds = get_dataset(
            file, 
            batch_size=batch_size,
            cache=cache,
            repeat=repeat,
            shuffle=shuffle,
            label_type=label_type,
            return_image_ids=return_image_ids,
            RGB=RGB)
        
        self.num_examples = counts_tfrecord(file)

        self.batch_size = batch_size
        self.return_image_ids = return_image_ids
        self._iterator = None
    
    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.ds)
        else:
            self._reset()
        return self._iterator

    def _reset(self):
        self._iterator = iter(self.ds)

    def __next__(self):
        batch = next(self._iterator)
        return batch

    def __len__(self):
        n_batches = self.num_examples // self.batch_size
        if self.num_examples % self.batch_size == 0:
            return n_batches
        else:
            return n_batches + 1


if __name__ == '__main__':

    sets = {
        'train': '/home/verlyndem/Documents/data/train.tfrecord',
        'val': '/home/verlyndem/Documents/data/val.tfrecord',
        'test': '/home/verlyndem/Documents/data/test.tfrecord'
    }

    BigEarthNetDataLoader(sets['test'], train=False, repeat=False, shuffle=False, label_type='original', return_image_ids=True)
    