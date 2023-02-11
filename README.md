Voice to Text Application (7 commands) 

This project is an Automatic Speech Recognition(ASR) application. Goal is to convert simple 7 voice commands to text.
Data set used in this project is https://www.tensorflow.org/datasets/catalog/speech_commands. CTC algorithm was used
convert voice to text.

Following steps were followed.

1) Load voice command data.
2) Convert voice command to spectrogram.
3) Convert to MEL spectrogram.
4) Apply LSTM and CTC Loss function to classify window to a character.
5) CTC greedy approach to derive the word.

Following tutorial was followed during the implementation.
https://www.tensorflow.org/tutorials/audio/simple_audio


