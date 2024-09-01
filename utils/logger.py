# import tensorflow as tf
import logging

#class Logger(object):
#    def __init__(self, log_dir):
#        """Create a summary writer logging to log_dir."""
#        self.writer = tf.summary.FileWriter(log_dir)
#
#    def scalar_summary(self, tag, value, step):
#        """Log a scalar variable."""
#        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
#        self.writer.add_summary(summary, step)
#
#    def list_of_scalars_summary(self, tag_value_pairs, step):
#        """Log scalar variables."""
#        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
#        self.writer.add_summary(summary, step)


def get_logger(file_path):

    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger
