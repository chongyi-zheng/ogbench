import os
from collections import defaultdict

import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('restore_dir', '', 'Dataset restore directory.')
flags.DEFINE_list('restore_dataset_names', [], 'Name of datasets to restore.')
flags.DEFINE_string('save_path', None, 'Save path.')


def main(_):
    train_dataset = defaultdict(list)
    val_dataset = defaultdict(list)
    for restore_dataset_name in FLAGS.restore_dataset_names:
        restore_path = os.path.join(FLAGS.restore_dir, restore_dataset_name + '.npz')
        assert os.path.exists(restore_path), \
            f"Restore path {restore_path} does not exist"
        assert os.path.exists(restore_path.replace('.npz', '-val.npz')), \
            f"Restore path {restore_path.replace('.npz', '-val.npz')} does not exist"

        train_path = restore_path
        val_path = restore_path.replace('.npz', '-val.npz')

        train_file = np.load(train_path)
        val_file = np.load(val_path)

        for k in ['observations', 'actions', 'rewards', 'terminals', 'masks']:
            train_dataset[k].append(train_file[k][...])

        for k in ['observations', 'actions', 'rewards', 'terminals', 'masks']:
            val_dataset[k].append(val_file[k][...])

    for k, v in train_dataset.items():
        train_dataset[k] = np.concatenate(v, axis=0)
    for k, v in val_dataset.items():
        val_dataset[k] = np.concatenate(v, axis=0)

    train_save_path = FLAGS.save_path
    val_save_path = FLAGS.save_path.replace('.npz', '_val.npz')

    for path, dataset in [(train_save_path, train_dataset), (val_save_path, val_dataset)]:
        np.savez_compressed(path, **dataset)


if __name__ == '__main__':
    app.run(main)
