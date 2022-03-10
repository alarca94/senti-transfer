from utils.in_out import *
from experiments import one_vs_rest


def perform_experiments():
    one_vs_rest.run('example.yml')


def gpu_usage_check():
    from torch import cuda
    print(f'Is CUDA available for Pytorch? {cuda.is_available()}')
    if cuda.is_available():
        print(f'Current CUDA device {cuda.current_device()}, {cuda.get_device_name(0)}, {cuda.device(0)}')
        print(f'How many GPU devices are available? {cuda.device_count()}')

    import warnings
    warnings.simplefilter('ignore')

    # import tensorflow as tf
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    # if tf.test.gpu_device_name():
    #     print(f'Default Tensorflow GPU Device: {tf.test.gpu_device_name()}')


if __name__ == '__main__':
    # gpu_usage_check()
    # prepare_files()
    perform_experiments()
