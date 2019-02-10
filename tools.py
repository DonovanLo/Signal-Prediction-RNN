#
# The methods in this script are credited to "Sequence to Sequence (seq2seq) Recurrent Neural Network (RNN) for Time Series Prediction" by Guillaume Chevalier.
#

import numpy as np

seed = 23
np.random.seed(seed)

initial_input_len = 20    # Input sequence length for encoder
initial_output_len = 20   # OUtput sequence length for decoder

num_signals = 2

fs = 30 
freq_delta = 20
freq_min = 0.1
amp_delta = 1
amp_min = 0.1

def genSine(epoch_step_size, batch_size, input_len=initial_input_len, output_len=initial_output_len):
    full_len = input_len + output_len
    t = np.arange(full_len)
    x = 2 * np.pi * t
    
    while True:
        for _ in range(epoch_step_size):
            for _ in range(num_signals):
                rand_freq = (np.random.rand(batch_size, 1) * (freq_delta + freq_min))
                rand_offset = np.random.rand(batch_size, 1) * 2 * np.pi
                rand_amp = (np.random.rand(batch_size, 1) * (amp_delta + amp_min))
                
                signals = rand_amp * np.sin(x*(rand_freq/fs) + rand_offset)
            signals = np.expand_dims(signals, axis=2)

            encoder_input = signals[:, :input_len, :]
            decoder_output = signals[:, input_len:, :]
            decoder_input = np.zeros((np.shape(decoder_output)[0], np.shape(decoder_output)[1], 1))
    
            yield ([encoder_input, decoder_input], decoder_output)

def genTwoFreq(epoch_step_size, batch_size, input_len=initial_input_len, output_len=initial_output_len):
    full_len = input_len + output_len
    t = np.arange(full_len)
    x = 2 * np.pi * t
    
    while True:
        for _ in range(epoch_step_size):
            for _ in range(num_signals):
                rand_freq = (np.random.rand(batch_size, 1) * (freq_delta + freq_min))
                rand_offset = np.random.rand(batch_size, 1) * 2 * np.pi
                rand_amp = (np.random.rand(batch_size, 1) * (amp_delta + amp_min))

                sine_signals = rand_amp * np.sin(x*(rand_freq/fs) + rand_offset)

                rand_freq = (np.random.rand(batch_size, 1) * (freq_delta + freq_min))
                rand_offset = np.random.rand(batch_size, 1) * 2 * np.pi
                rand_amp = (np.random.rand(batch_size, 1) * (amp_delta + amp_min))

                signals = rand_amp * np.cos(x*(rand_freq/fs) + rand_offset) + sine_signals

            signals = np.expand_dims(signals, axis=2)

            encoder_input = signals[:, :input_len, :]
            decoder_output = signals[:, input_len:, :]
            decoder_input = np.zeros((np.shape(decoder_output)[0], np.shape(decoder_output)[1], 1))

            yield ([encoder_input, decoder_input], decoder_output)

def genTwoFreq_noise(epoch_step_size, batch_size, input_len=initial_input_len, output_len=initial_output_len):
    full_len = input_len + output_len
    t = np.arange(full_len)
    x = 2 * np.pi * t
    
    while True:
        for _ in range(epoch_step_size):
            for _ in range(num_signals):
                rand_freq = (np.random.rand(batch_size, 1) * (freq_delta + freq_min))
                rand_offset = np.random.rand(batch_size, 1) * 2 * np.pi
                rand_amp = (np.random.rand(batch_size, 1) * (amp_delta + amp_min))

                sine_signals = rand_amp * np.sin(x*(rand_freq/fs) + rand_offset)

                rand_freq = (np.random.rand(batch_size, 1) * (freq_delta + freq_min))
                rand_offset = np.random.rand(batch_size, 1) * 2 * np.pi
                rand_amp = (np.random.rand(batch_size, 1) * (amp_delta + amp_min))

                signals = rand_amp * np.cos(x*(rand_freq/fs) + rand_offset) + sine_signals
                
                noise_amount = np.random.random() * 0.15
                signals = signals + noise_amount * np.random.rand(batch_size, 1) 
                
                avg = np.average(signals)
                std = np.std(signals) + 0.0001
                signals = 2.5 * ( signals - avg ) / std
                
            signals = np.expand_dims(signals, axis=2)

            encoder_input = signals[:, :input_len, :]
            decoder_output = signals[:, input_len:, :]
            decoder_input = np.zeros((np.shape(decoder_output)[0], np.shape(decoder_output)[1], 1))

            yield ([encoder_input, decoder_input], decoder_output)

genSignal = {
	1: genSine,
	2: genTwoFreq,
	3: genTwoFreq_noise
}


from matplotlib import pyplot as plt

def plot_prediction(x, y_true, y_pred):
    """Plots the predictions.
    
    Arguments
    ---------
    x: Input sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_true: True output sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_pred: Predicted output sequence (input_sequence_length,
        dimension_of_signal)
    """

    plt.figure(figsize=(12, 3))

    output_dim = x.shape[-1]
    for j in range(output_dim):
        past = x[:, j] 
        true = y_true[:, j]
        pred = y_pred[:, j]

        label1 = "Seen (past) values" if j==0 else "_nolegend_"
        label2 = "True future values" if j==0 else "_nolegend_"
        label3 = "Predictions" if j==0 else "_nolegend_"

        plt.plot(range(len(past)), past, "o--b",
                 label=label1)
        plt.plot(range(len(past),
                 len(true)+len(past)), true, "x--b", label=label2)
        plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y",
                 label=label3)
    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()
