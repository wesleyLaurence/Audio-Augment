# Audio-Augment
### Audio data augmentation tool for machine learning projects
This library augments audio training datasets by transforming provided WAV files in 10 different ways:
        
  1. Waveform Inversion
  2. Highpass Filter
  3. Lowpass Filter
  4. Bandpass Filter
  5. Add noise (normal, uniform)
  6. Pitch shift (low, high)
  7. Time shift (slow, fast)
   
## Directory Setup
Create a folder in your working directory called 'WAV'. Then create two subfolders inside WAV, one called 'unprocessed' and another called 'processed'.
Place all audio samples that you want to transform in the unprocessed folder (must be 16-bit/44.1kHz WAV format). 
Run the `AudioAugment()` method and it will transform all audio files in your unprocessed folder, 
and save the new set of augmented samples in the processed folder.

## Installation
`pip install AudioAugment`

## Requirements
- numpy
- pandas
- matplotlib
- soundfile
- librosa

<b>For questions about this library, email wesleylaurencetech@gmail.com</b>
