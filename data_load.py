import tensorflow as tf
import re
from string import ascii_lowercase, digits
import numpy as np
import os


class DataLoader:

    def __generate_label(self, text):
        space_token = ' '
        blank_token = '$'
        # alphabet = list(ascii_lowercase) + [space_token, end_token, blank_token, decimal_point] + list(digits)
        alphabet = list(ascii_lowercase) + list(digits) + [space_token, blank_token]
        char_index_dic = {}
        for index, char in enumerate(alphabet):
            char_index_dic[char] = index
        outputs = []
        punctuation_removed = re.sub(r'([^\w\s]|_)', '', text)
        for char in punctuation_removed.lower():
            outputs.append(char_index_dic[char])
        return tf.convert_to_tensor(outputs)

    def __get_dir_name(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)

        # Note: You'll use indexing here instead of tuple unpacking to enable this
        # to work in a TensorFlow graph.
        return parts[-2]

    def load_mini_speech_commands(self, audio_dir_path):
        if not audio_dir_path.exists():
            tf.keras.utils.get_file(
                'mini_speech_commands.zip',
                origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                extract=True,
                cache_dir='.', cache_subdir='data')
        commands = np.array(tf.io.gfile.listdir(str(audio_dir_path)))
        commands = commands[commands != 'README.md']
        print('Commands:', commands)
        file_names = tf.io.gfile.glob(str(audio_dir_path) + '/*/*')
        file_names = tf.random.shuffle(file_names)
        num_samples = len(file_names)
        label_list = []
        for file in file_names:
            label_txt = self.__get_dir_name(file)
            label = self.__generate_label(str(label_txt.numpy(), "utf-8"))
            label_list.append(label)
        label_np_array = tf.keras.preprocessing.sequence.pad_sequences(label_list, value=36, padding="post")
        label_set = tf.convert_to_tensor(label_np_array)
        print('Number of total examples:', num_samples)
        print('Number of examples per label:',
              len(tf.io.gfile.listdir(str(audio_dir_path / commands[0]))))
        print('Example file tensor:', file_names[0])
        return file_names, label_set

