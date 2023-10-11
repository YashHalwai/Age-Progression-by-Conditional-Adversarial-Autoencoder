import tensorflow as tf
from FaceAging import FaceAging
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='CAAE')
parser.add_argument('--is_train', type=str2bool, default=True)
parser.add_argument('--epoch', type=int, default=1, help='number of epochs')
parser.add_argument('--dataset', type=str, default='UTKFace', help='training dataset name that stored in ./data')
parser.add_argument('--savedir', type=str, default='save', help='dir of saving checkpoints and intermediate training results')
parser.add_argument('--testdir', type=str, default='None', help='dir of testing images')
parser.add_argument('--use_trained_model', type=str2bool, default=True, help='whether to train from an existing model or from scratch')
parser.add_argument('--use_init_model', type=str2bool, default=True, help='whether to train from the init model if cannot find an existing model')
FLAGS = parser.parse_args()

def main(_):
    # Print settings
    import pprint
    pprint.pprint(FLAGS)

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    with tf.compat.v1.Session() as session:
        model = FaceAging(
            session,
            is_training=FLAGS.is_train,
            save_dir=FLAGS.savedir,
            dataset_name=FLAGS.dataset
        )
        if FLAGS.is_train:
            print('\nTraining Mode')
            if not FLAGS.use_trained_model:
                print('\nPre-train the network')
                model.pretrain(
                    num_epochs=10,
                    use_init_model=FLAGS.use_init_model,
                    weights=(0, 0, 0)
                )
                print('\nPre-train is done! The training will start.')
            model.train(
                num_epochs=FLAGS.epoch,
                use_trained_model=FLAGS.use_trained_model,
                use_init_model=FLAGS.use_init_model
            )
        else:
            print('\nTesting Mode')
            model.custom_test(
                testing_samples_dir=FLAGS.testdir + '/*jpg'
            )

if __name__ == '__main__':
    tf.compat.v1.app.run()