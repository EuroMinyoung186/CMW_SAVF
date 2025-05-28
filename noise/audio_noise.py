import torch
import numpy as np
from scipy.signal import butter, lfilter
from pydub import AudioSegment
from io import BytesIO

import torch

def add_uniform_noise(signal, desired_snr_db = 34.5):
    """
    Adds uniformly distributed noise to an audio signal to achieve the desired SNR.

    Parameters:
        signal (torch.Tensor): The original audio signal tensor.
        desired_snr_db (float): The desired Signal-to-Noise Ratio in decibels.

    Returns:
        torch.Tensor: The noisy audio signal.
    """
    # Calculate the power of the original signal
    signal_power = torch.mean(signal ** 2)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (desired_snr_db / 10)

    # Calculate the required noise power
    noise_power = signal_power / snr_linear

    # For a uniform distribution in the range [-a, a], the variance is (a^2) / 3
    # Solve for 'a' to match the noise power (variance)
    noise_std = torch.sqrt(3 * noise_power)

    # Generate uniform noise in the range [-a, a]
    noise = torch.rand_like(signal) * 2 * noise_std - noise_std

    # Add the noise to the original signal
    noisy_signal = signal + noise

    return noisy_signal

import torch

def reduce_amplitude(signal, scaling_factor=0.9):
    """
    Reduces the amplitude of the audio signal to a specified percentage of the original.

    Parameters:
        signal (torch.Tensor): The original audio signal tensor.
        scaling_factor (float): The factor by which to scale the amplitude (default is 0.9).

    Returns:
        torch.Tensor: The amplitude-scaled audio signal.
    """
    # Scale the amplitude of the signal
    scaled_signal = signal * scaling_factor
    return scaled_signal


import torch

def add_echo(signal, sample_rate, delay_ms=100, attenuation=0.3):
    """
    Adds an echo to the audio signal by attenuating it, delaying it, and overlaying it with the original.

    Parameters:
        signal (torch.Tensor): The original audio signal tensor.
        sample_rate (int): The sampling rate of the audio signal in Hz.
        delay_ms (float): The delay of the echo in milliseconds (default is 100ms).
        attenuation (float): The attenuation factor for the echo (default is 0.3).

    Returns:
        torch.Tensor: The audio signal with the echo added.
    """
    # Convert delay from milliseconds to number of samples
    delay_samples = int(sample_rate * delay_ms / 1000)

    # Create an empty tensor for the delayed signal
    delayed_signal = torch.zeros_like(signal)

    if delay_samples < len(signal):
        # Attenuate the original signal
        attenuated_signal = signal * attenuation

        # Overlay the attenuated signal onto the delayed signal
        delayed_signal[delay_samples:] = attenuated_signal[:-delay_samples]

    # Add the delayed, attenuated signal to the original signal
    output_signal = signal + delayed_signal

    return output_signal

# Example usage:
# Load your audio signal as a PyTorch tensor
# audio_signal = torch.load('your_audio_tensor.pt')
# sample_rate = 44100  # Replace with your actual sample rate
# echo_audio = add_echo(audio_signal, sample_rate, delay_ms=100, attenuation=0.3)

