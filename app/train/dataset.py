import tensorflow as tf
import tensorflow_datasets as tfds

def get_dataset(split='train'):
    """Download and load a TensorFlow dataset.
    Args:
        split: The dataset split to load ('train', 'test').

    Returns:
        A TensorFlow dataset object.
    """
    
    # Load the dataset
    ds, ds_info = tfds.load(
        "stanford_dogs",
        split=split,
        as_supervised=True,
        with_info=True,
        shuffle_files=True
    )

    return ds, ds_info


def prepare_dataset(dataset, config):
    """Preprocess the dataset according to the given configuration.

    Args:
        dataset: A TensorFlow dataset to preprocess.
        config: A dictionary containing preprocessing parameters.

    Returns:
        A preprocessed TensorFlow dataset.
    """
    # Example preprocessing steps based on config
    if config.get("shuffle", False):
        dataset = dataset.shuffle(buffer_size=config.get("shuffle_buffer_size", 1000))
    
    if config.get("batch_size"):
        dataset = dataset.batch(config["batch_size"])
    
    if config.get("prefetch", False):
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def data_preprocess(dataset, config_path='config/preprocess_config.yaml'):
    """Load preprocessing configuration and apply it to the dataset.

    Args:
        dataset: A TensorFlow dataset to preprocess.
        config_path: Path to the YAML configuration file.

    Returns:
        A preprocessed TensorFlow dataset.
    """

    # Remove additional channels
    if single_channel: image = tf.expand_dims(image[:, :, 0], axis=-1)
    
    # Resize video
    if resize_shape: image = tf.image.resize(image, resize_shape)
    
    # Label encoding
    label_encoded = tf.squeeze(tf.one_hot(label, num_classes))

    # Normalization: [0, 255] => [-1, 1] floats
    image = tf.cast(image, tf.float32) * (1./127.5)-1 # (1./255.)
        
    # Training / Not training stage
    if stage == 'train': 
        
        # Add random brightness
        if augmentation['brightness']:
            image = tf.image.random_brightness(image, augmentation['brightness'])
                
        # Add random contrast
        if augmentation['contrast']:
            image = tf.image.random_contrast(image, augmentation['contrast'][0],
                                             augmentation['contrast'][1])
        
        # Add gaussian noise
        # noise = tf.random.normal(shape=tf.shape(image)[:-1], mean=0, stddev=0.02, dtype=tf.float32)
        # noise = tf.tile(tf.expand_dims(noise, -1), [1,1,3])
        # image = image + noise
        
        # Consistent left_right_flip for entire video
        if augmentation['flip']:
            flip_random = tf.random.uniform(shape=[], minval=0, maxval=1, 
                                            dtype=tf.float32)
            option = tf.less(flip_random, 0.5)
            image = tf.cond(option,
                              lambda: tf.image.flip_left_right(image),
                              lambda: tf.identity(image))
        
        # Adapt to the range [-1, 1]
        image = tf.clip_by_value(image, -1, 1)
         
    return image, label_encoded, path # Remove for training