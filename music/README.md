# Timbre transfer with the pre-trained models

Please check *inference.ipynb* for the detailed instructions.

The HiFi-GAN vocoder trained on our music dataset: https://drive.google.com/file/d/1PN2zWlo4PmxZ5EYLvJ6zGyJjrgrJN2Pg/view?usp=sharing.  

The diffusion-based music generator model: https://drive.google.com/file/d/10GOAFPjTx4by89WPO3MmCGdXCXoIimRN/view?usp=sharing.

Classifier for controlled inference: https://drive.google.com/file/d/14uww_vIy3WTsjVYYk6aHV3uDHPHUN9Tu/view?usp=sharing.

Pitch extractor for controlled inference: https://drive.google.com/file/d/114wllriiCaJuHXFA9_cROi0q0a2kacIX/view?usp=sharing.

Please put these four models to *checkpts/*.

# Training your own model

0. To train model on your data, first create a data directory with three folders: "wavs", "mels" and "pitch". Put raw audio files sampled at 22.05kHz to "wavs" directory. The function for calculating mel-spectrograms can be found at *inference.ipynb* notebook (*get_mel*). The notebook *get_pitch.ipynb* shows how to extract pitch from wav file.

1. Create "logs" directory and run *train_generator.py* to train diffusion-based music generator. If you want to perform gradient-guided sampling, then you also have to run *train_classifier.py* and *train_pitch_extractor.py*.

2. Please check *params.py* for the most important hyperparameters.
