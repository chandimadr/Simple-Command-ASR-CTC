import tensorflow as tf
from string import ascii_lowercase, digits
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from data_load import DataLoader
import configparser


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_waveform_and_label(file_path, label):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    if "mini_speech_command" in data_source_list:
        time_factor = 1
    zero_padding = tf.zeros([16000*time_factor] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    stfts = tf.signal.stft(equal_length, frame_length=frame_length, frame_step=frame_step)

    spectrogram = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, 16000
                                                                        , lower_edge_hertz, upper_edge_hertz)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    return log_mel_spectrogram


def plot_signal(data_set):
    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
    for i, (audio, label) in enumerate(data_set.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(audio.numpy())
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        # label = label.numpy()
        ax.set_title(i)

    plt.show()


def get_spectrogram_and_label(audio, label):
    spectrogram = get_spectrogram(audio)
    # spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram, label


def plot_spectrogram_graph(data_set):
    rows = 3
    cols = 4
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, (spectrogram, label) in enumerate(data_set.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
        ax.set_title(label[:].numpy())
        ax.axis('off')

    plt.show()


def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


class CTCLoss(tf.keras.losses.Loss):

    def __init__(self, logits_time_major=False, blank_index=-1,
                 name='ctc_loss'):
        super().__init__(name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def call(self, y_true, y_pred):

        y_true_new = tf.cast(y_true, tf.int32)
        y_pred_shape = tf.shape(y_pred)
        logit_length = tf.fill([y_pred_shape[0]], y_pred_shape[1])
        y_true_shape = tf.shape(y_true_new)
        label_length = tf.fill([y_true_shape[0]], y_true_shape[1])
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index,
            unique=None
        )
        return tf.math.reduce_mean(loss)


def get_data_set(files, labels):
    files_ds = tf.data.Dataset.from_tensor_slices((files, labels))
    waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    plot_signal(waveform_ds)

    spectrogram_ds = waveform_ds.map(get_spectrogram_and_label, num_parallel_calls=AUTOTUNE)

    # plot_spectrogram_graph(spectrogram_ds)
    return spectrogram_ds


def decode_seq(seq_out):
    # greedy decoding
    space_token = ' '
    # end_token = '>'
    blank_token = '$'
    alphabet = list(ascii_lowercase) + list(digits) + [space_token, blank_token]
    output_text = []

    for seq in seq_out:
        out = ''
        for timestep in seq:
            out += alphabet[tf.math.argmax(timestep)]
        output_text.append(out)
    return output_text


def decode_tgt(tgt_seq):
    # greedy decoding
    target = tgt_seq[:].numpy()
    space_token = ' '
    # end_token = '>'
    blank_token = '$'
    alphabet = list(ascii_lowercase) + list(digits) + [space_token, blank_token]
    output_text = []

    for seq in target:
        out = ''
        for letter in seq:
            if letter < len(alphabet):
                out += alphabet[letter]
        output_text.append(out)
    return output_text


class LossExceedCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') > 500000:
            print("\nModel is going wrong way!")
            self.model.stop_training = True


class MinLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') <= 1:
            print("\nModel is GOOD!!!.. Stopping training more")
            self.model.stop_training = True


def detokenize(table, x):
    x = tf.RaggedTensor.from_sparse(x)
    x = tf.ragged.map_flat_values(table.lookup, x)
    strings = tf.strings.reduce_join(x, axis=1)
    return strings


if __name__ == "__main__":
    AUTOTUNE = tf.data.AUTOTUNE

    config = configparser.ConfigParser()
    config.read('asr_config.properties')

    EPOCHS = int(config['Main']['epochs'])
    batch_size = int(config['Main']['batch_size'])
    decoder_type = ['softmax', 'ctc_greedy']        # softmax, beam_search, ctc_greedy

    frame_length = int(config['Main']['frame_length'])
    frame_step = int(config['Main']['frame_step'])

    data_loader = DataLoader()
    data_source_list = ["mini_speech_command"]
    if "mini_speech_command" in data_source_list:
        print("*************** Mini Commands Data Set ***************")
        data_dir = pathlib.Path('data/mini_speech_commands')
        train_samples = int(config['mini_speech_command']['train_samples'])
        val_samples = int(config['mini_speech_command']['val_samples'])
        test_samples = int(config['mini_speech_command']['test_samples'])
        file_names, label_set = data_loader.load_mini_speech_commands(data_dir)

    train_files = file_names[:train_samples]
    val_files = file_names[train_samples: train_samples+val_samples]
    test_files = file_names[-test_samples:]

    train_label_files = label_set[:train_samples]
    val_label_files = label_set[train_samples: train_samples+val_samples]
    test_label_files = label_set[-test_samples:]

    train_ds = get_data_set(train_files, train_label_files)
    validation_ds = get_data_set(val_files, val_label_files)
    test_ds = get_data_set(test_files, test_label_files)

    # plot_spectrogram_graph(train_ds)

    for spectrogram, _ in train_ds.take(1):
        input_shape = spectrogram.shape
    print('##Input shape:', input_shape)

    train_ds = train_ds.batch(batch_size)
    validation_ds = validation_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    loss_exceed_cb = LossExceedCallback()
    min_loss_cb = MinLossCallback()

    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(inputs)

    forward_layer = tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True)
    backward_layer = tf.keras.layers.LSTM(128, activation='relu', return_sequences=True, go_backwards=True)
    x = tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(x)
    outputs = tf.keras.layers.Dense(38)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="asr_model")
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=CTCLoss(),
        # metrics=[SequenceAccuracy()],
    )

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=[loss_exceed_cb, min_loss_cb]
    )

    predict_out = model.predict(test_ds)

    init = tf.lookup.TextFileInitializer(
        filename="label_table.txt",
        key_dtype=tf.int64, key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        value_dtype=tf.string, value_index=tf.lookup.TextFileIndex.WHOLE_LINE)
    char_table = tf.lookup.StaticHashTable(init, '')

    if 'softmax' in decoder_type:
        softmax_out = tf.nn.log_softmax(predict_out)
        predicted_txt = decode_seq(softmax_out)
        tgt_text = decode_tgt(test_label_files)
        print("---------- Softmax Decoding ----------")
        for i, ele in enumerate(predicted_txt):
            print(ele, "::", tgt_text[i])

    if 'ctc_greedy' in decoder_type:
        pred_out_shape = tf.shape(predict_out)
        sequence_length = tf.fill([pred_out_shape[0]], pred_out_shape[1])
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
            tf.transpose(predict_out, perm=[1, 0, 2]),
            sequence_length,
            merge_repeated=True)
        tgt_strings = detokenize(char_table, x=decoded[0])
        greedy_list = tgt_strings.numpy().astype('str')
        print("---------- CTC Greedy Decoding ----------")
        for i, ele in enumerate(greedy_list):
            print(ele, "::", tgt_text[i])
