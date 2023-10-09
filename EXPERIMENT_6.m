QUESTION 1

% Given parameters
alpha = 1+mod(259,3);           % Maximum passband ripple in dB
stopband_attenuation = 40; % Minimum stopband attenuation in dB
Fs = 720;            % Sampling frequency in samples/sec
Fp = 10;             % Passband edge frequency in Hz
Fstop = 20;          % Stopband edge frequency in Hz

% Calculate normalized cutoff frequency (Wc)
Wc = 2 * Fp / Fs;
Ws= 2*Fstop/Fs;
% Determine filter order (N) using the formula for Butterworth filter
[n,Wn] = buttord(Wc,Ws,alpha,40) 
% Design the Butterworth filter
[z,p] = butter(n,Wn);
sos = tf(z,p,1/Fs);
pzmap(sos)
% Plot the pole-zero plot
figure;
pzmap(sos);
title('Pole-Zero Plot');
grid on;
% Plot the Bode plot
figure;
bode(sos);
title('Bode Plot');
grid on;
% Define the time vector for 1 second
t = 0:1/Fs:1;

% Compute the impulse response of the filter
impulse_response = impulse(sos, t);

% Compute the step response of the filter
step_response = step(sos, t);

% Plot the impulse response and step response on the same graph
figure;
plot(t, impulse_response, 'b', 'LineWidth', 2, 'DisplayName', 'Impulse Response');
hold on;
plot(t, step_response, 'r', 'LineWidth', 2, 'DisplayName', 'Step Response');
xlabel('Time (s)');
ylabel('Amplitude');
title('Impulse Response and Step Response');
legend('show');
grid on;
n =

     8

Wn =

    0.0313

QUESTION 2
% Filter design parameters
alpha = 2;        % Maximum passband ripple (in dB)
A = 40;             % Minimum stopband attenuation (in dB)
f_pass = 10;        % Passband edge frequency (in Hz)
f_stop = 20;        % Stopband edge frequency (in Hz)
fs = 720;           % Sampling frequency (in samples/sec)
ecg_data=load('ECG_Data.txt')
% Calculate the minimum filter order using butterord
[N, Wn] = buttord(f_pass / (fs/2), f_stop / (fs/2), alpha, A);

% Design the Butterworth filter
[b, a] = butter(N, Wn, 'low');

% Apply the Butterworth filter to the ECG data (assuming you have loaded the data)
filtered_ecg = filtfilt(b, a, ecg_data);

% Create a time vector for plotting
t = (0:length(ecg_data) - 1) / fs;

% Plot the original and filtered signals in the same figure
figure;
subplot(2, 1, 1);
plot(t, ecg_data);
title('Original ECG Signal');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2, 1, 2);
plot(t, filtered_ecg);
title('Filtered ECG Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Adjust the plot layout
sgtitle('ECG Signal and Filtered Signal Comparison');

% Optionally, save the filtered ECG data to a text file
% save('filtered_ecg.txt', 'filtered_ecg', '-ascii');


QUESTION 3
% Load the audio file
filename = 'instru2.wav';
[y, Fs] = audioread(filename);

% Compute the spectrogram of the audio signal
window_size = 1024;  % You can adjust this based on your preference
overlap = 512;       % Adjust overlap as needed
nfft = 2048;         % Adjust FFT size as needed

spectrogram(y, hamming(window_size), overlap, nfft, Fs, 'yaxis');
title('Original Spectrogram');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

% Design a digital Butterworth bandpass filter
f_passband = [100 1000];  % Passband frequency range (adjust as needed)
order = 4;                % Filter order (adjust as needed)

[b, a] = butter(order, f_passband / (Fs/2), 'bandpass');

% Apply the filter to the audio signal
filtered_signal = filter(b, a, y);

% Save the filtered audio to a new WAV file
output_filename = 'filtered_instruα.wav';
audiowrite(output_filename, filtered_signal, Fs);

% Plot the spectrogram of the filtered audio
figure;
spectrogram(filtered_signal, hamming(window_size), overlap, nfft, Fs, 'yaxis');
title('Filtered Spectrogram');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;

% Play the filtered audio
sound(filtered_signal, Fs);


QUESTION 4
% Given specifications
alpha = 2;               % Maximum passband ripple in dB
stopband_attenuation = 40; % Minimum stopband attenuation in dB
cutoff_frequency = 10;     % Cutoff frequency in Hz
sampling_frequency = 720;  % Sampling frequency in samples/sec

% Calculate normalized cutoff frequency (Wc)
Wc = 2 * cutoff_frequency / sampling_frequency;

% Calculate the order of the Chebyshev Type I filter (N)
N = ceil(log10(((10^(-alpha/10) - 1) / (10^(-stopband_attenuation/10) - 1)) / ((0.1^2) - 1)) / (2 * log10(Wc)));

% Design the Chebyshev Type I filter
[b, a] = cheby1(N, alpha, Wc);

% Normalize the filter coefficients
b = b / b(1);

% Frequency response for Chebyshev Type I filter
[h, w] = freqz(b, a, 8000, 'whole', sampling_frequency);

% Plot the magnitude response (Bode plot)
figure;
semilogx(w, 20*log10(abs(h)));
xlabel('Frequency (Hz)');
ylabel('Gain (dB)');
title('Bode Plot of Chebyshev Type I Filter');
grid on;

% Generate impulse response for Chebyshev Type I filter
impulse_response_chebyshev = filter(b, a, [1 zeros(1, 999)]);

% Generate step response for Chebyshev Type I filter
step_response_chebyshev = cumsum(impulse_response_chebyshev);

% Generate impulse and step response for Butterworth filter (use the coefficients from Problem 1)
% Replace these coefficients with the ones you obtained in Problem 1.
% Example: b_butter = [1, -1.2369, 1.4464, -0.9745, 0.8211, 0.3531];
%          a_butter = [1, -2.0192, 1.5610, -0.8547, 0.1453];
b_butter = [1]; % Replace with Butterworth coefficients
a_butter = [1]; % Replace with Butterworth coefficients
impulse_response_butter = filter(b_butter, a_butter, [1 zeros(1, 999)]);
step_response_butter = cumsum(impulse_response_butter);

% Plot impulse response
figure;
plot(impulse_response_chebyshev, 'DisplayName', 'Chebyshev Type I');
hold on;
plot(impulse_response_butter, 'DisplayName', 'Butterworth');
xlabel('Time (samples)');
ylabel('Amplitude');
title('Impulse Response Comparison');
grid on;
legend;

% Plot step response
figure;
plot(step_response_chebyshev, 'DisplayName', 'Chebyshev Type I');
hold on;
plot(step_response_butter, 'DisplayName', 'Butterworth');
xlabel('Time (samples)');
ylabel('Amplitude');
title('Step Response Comparison');
grid on;
legend;