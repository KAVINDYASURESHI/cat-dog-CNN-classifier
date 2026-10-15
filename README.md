# Dog vs. Cat Image Classifier (CNN)

A convolutional neural network pipeline for classifying dog and cat images: data preprocessing, model definition, training on Google Colab, and a test notebook with folder and live-webcam inference.

> Status: work in progress. The training pipeline is complete; I'm now improving how well the model generalizes to unseen images.

## Dataset
Kaggle Cats and Dogs (Microsoft PetImages): 24,997 usable images (3 unreadable files skipped). Not included in this repository. Download it from Kaggle and set `data_path` in `data_preprocesing.ipynb` to where you extracted `PetImages`.

## Pipeline
1. `data_preprocesing.ipynb`: reads images with OpenCV, resizes to 50x50, converts to grayscale, normalizes to [0, 1], one-hot encodes labels, saves `data.npy` and `target.npy`
2. `CNN-model-training-colab.ipynb`: defines and trains the CNN on Google Colab with augmentation and checkpointing
3. `CNN-model-test.ipynb`: loads saved weights and runs inference on a folder of images or a live webcam feed

Model weights and `test_images/` are not included. Train your own with notebook 2.

## Model
- 3 convolutional blocks: Conv2D (256 → 128 → 64 filters, 3x3), each followed by BatchNormalization, ReLU and 2x2 MaxPooling
- Flatten → Dropout 0.5 → Dense 200 → Dropout 0.5 → Dense 100 → Dense 50 → Dense 2 (softmax); 603,436 parameters
- Adam optimizer, categorical cross-entropy
- Augmentation: rotation, zoom, width/height shift, shear, horizontal flip
- 90/10 train/test split, 20 epochs, best checkpoints saved by validation loss

## Tech Stack
Python · TensorFlow/Keras · OpenCV · NumPy · scikit-learn · Matplotlib · Google Colab · Jupyter Notebook

## Author
Kavindya Hewage | Abu Dhabi, UAE
LinkedIn: linkedin.com/in/kavindya-sureshi-8b39681b8/
Portfolio: https://KAVINDYASURESHI.github.io
