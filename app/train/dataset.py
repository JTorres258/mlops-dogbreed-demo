import tensorflow as tf
import tensorflow_datasets as tfds
import os
from functools import partial
from config import load_config, DataConfig

os.chdir(os.path.dirname(os.path.realpath(__file__)))


def get_dataset(split="train"):
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
        shuffle_files=True,
    )

    return ds, ds_info


def prepare_dataset(
    dataset: tf.data.Dataset,
    config: DataConfig,
    *,
    training: bool,
    deterministic: bool | None = None,
) -> tf.data.Dataset:
    """
    Config-driven tf.data pipeline that applies your data_preprocessing function.
    Order: map -> cache -> (shuffle if train) -> batch -> prefetch.
    Args:
        dataset: A TensorFlow dataset to preprocess.
        config: A DataConfig object containing preprocessing parameters.
        training: Whether the dataset is for training (enables shuffling).
        deterministic: If set, controls the determinism of the dataset pipeline.
    Returns:
        A preprocessed TensorFlow dataset.
    """

    # Optional determinism control (throughput vs. reproducibility)
    if deterministic is not None:
        opts = tf.data.Options()
        opts.experimental_deterministic = bool(deterministic)
        dataset = dataset.with_options(opts)

    # Preprocessing mapping
    map_fn = partial(data_preprocessing, config=config, training=training)
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache if enabled (great for small/medium datasets or repeated epochs)
    if config.cache:
        dataset = dataset.cache()

    # Shuffle only for training
    if training and config.shuffle:
        dataset = dataset.shuffle(buffer_size=config.shuffle_buffer_size)

    # Batch (drop_remainder if you need static shapes)
    dataset = dataset.batch(config.batch_size, drop_remainder=config.drop_remainder)

    # Prefetch to overlap CPU prep & device compute
    if config.prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def data_preprocessing(
    image: tf.Tensor, label: tf.int64, *, config: DataConfig, training: bool
):
    """Data preprocessing function applying transformations based on config.
    Args:
        image: Input image tensor.
        label: Corresponding label tensor.
        config: DataConfig object with preprocessing parameters.
        training: Boolean indicating if in training mode (enables augmentations).
    Returns:
        Preprocessed image and label tensors.
    """

    # Ensure image has rank 3
    image = tf.ensure_shape(image, [None, None, None])  # helps shape inference

    # Resize video
    if config.resize:
        image = tf.image.resize(image, config.resize)

    # Normalization: [0, 255] => [0, 1] floats
    if config.normalize:
        image = tf.cast(image, tf.float32) * (1.0 / 255.0)
        if config.dtype == "float16":
            image = tf.cast(image, tf.float16)

    # Augmentation
    if training and config.augment:
        aug = config.augment
        # While deifining an if statement per augmentation is enough, here
        # we use a more defensive style that provides a default value if some
        # attributes are missing in the config object. Just as an example.
        if getattr(aug, "flip_left_right", False):
            image = tf.image.random_flip_left_right(image)
        if getattr(aug, "flip_up_down", False):
            image = tf.image.random_flip_up_down(image)
        if getattr(aug, "random_crop", False):
            crop_size = aug.random_crop
            image = tf.image.random_crop(
                image, size=[crop_size[0], crop_size[1], tf.shape(image)[-1]]
            )

    # Labels usually don't need modification
    label = tf.cast(label, tf.int32)

    return image, label


if __name__ == "__main__":
    # Example usage
    config = load_config("./configs/config_test.yml")
    train_ds, ds_info = get_dataset(split="train")
    train_ds_prepared = prepare_dataset(train_ds, config, training=True)
    for batch in train_ds_prepared.take(1):
        images, labels = batch
        print(tf.math.reduce_max(images), tf.math.reduce_min(images))
        print(images.shape, labels.shape)
        print(labels)
