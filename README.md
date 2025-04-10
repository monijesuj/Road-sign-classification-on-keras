# Road-sign-classification-on-keras

## Overview

This project is aimed at detecting and classifying road signs using deep learning with Keras and TensorFlow. The system receives images from cameras, processes them to detect and classify road signs, and then controls a robotic vehicle based on the classification. In addition, the project integrates sensor data to ensure safe operations when obstacles are detected.

## Project Structure

```
/data          - CSV files and other raw data for training and testing (e.g. Meta.csv, Test.csv, Train.csv)
/models        - Saved model files and weights (e.g. mobilenet_model.h5, model.h5, modela.h5, SignRec-CNN.h5, mobilenet.tflite, mobilenetv3_model.h5, weights.h5)
/notebooks     - Jupyter notebooks for experimentation and prototyping (e.g. BotProject.ipynb)
/src           - Python source files for training, evaluation, and inference (e.g. ann.py, chatpi.py, convert.py, modelchange.py, Training.py, traintrain.py, trainv5.py)
README.md      - This documentation file
```

## Getting Started

### Prerequisites

- Python 3.6 or higher
- [TensorFlow](https://www.tensorflow.org/) (which provides the Keras API)
- OpenCV
- NumPy, Pandas, Matplotlib, and Seaborn

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/monijesuj/Road-sign-classification-on-keras.git
   cd Road-sign-classification-on-keras
   ```

2. **Create a virtual environment and install dependencies:**

   Create a virtual environment:

   ```sh
   python -m venv venv
   source venv/bin/activate      # On Windows use `venv\Scripts\activate`
   ```

    Requirements:

   ```txt
   tensorflow>=2.0
   opencv-python
   numpy
   pandas
   matplotlib
   seaborn
   ```


## Folder Organization

- **/src:** Contains all source code for training, testing, and inference.
  - [`chatpi.py`](src/chatpi.py): Preprocesses images and runs model training and evaluation.
  - [`ann.py`](src/ann.py): Serves as an entry point for model inference.
  - [`convert.py`](src/convert.py): Responsible for conversion tasks of models.
  - [`modelchange.py`](src/modelchange.py): Contains legacy TensorFlow v1 code for model conversion.
  - Training scripts (`Training.py`, `traintrain.py`, `trainv5.py`): Explore different training configurations and architectures.
- **/models:** Stores all models and weights generated throughout the project.
- **/data:** Holds CSV files and any additional data required for model training or evaluation.
- **/notebooks:** Contains Jupyter notebooks (e.g., [`BotProject.ipynb`](notebooks/BotProject.ipynb)) used for prototyping, visualization, and experiments.

## Usage

### Data Preparation

Place your training and testing data (images and labels) in the appropriate locations. The project uses CSV files and a folder of images to create datasets. Check the data loading sections in [`chatpi.py`](src/chatpi.py) for guidance on file structure.

### Training

Adjust parameters such as image size, batch size, and epochs in the training scripts in `/src`. For instance, [`chatpi.py`](src/chatpi.py) uses a batch size of 64 and resizes images to 128×128 for training.

Run the training script:

```sh
python src/chatpi.py
```

Alternatively, you can use the provided Jupyter Notebook:

```sh
jupyter notebook notebooks/BotProject.ipynb
```

### Evaluation

After training, evaluate your model using the test datasets. The scripts print metrics such as accuracy and loss after evaluation. Refer to the training scripts and notebook cells for evaluation commands.

### Model Conversion

For model conversion (e.g., converting to TensorFlow Lite), use [`trainv5.py`](src/trainv5.py). Legacy conversion code exists in [`modelchange.py`](src/modelchange.py) for frozen graph conversion (TensorFlow v1) - remove this if you rely solely on TensorFlow v2 workflows.

### Inference

For inference, use [`ann.py`](src/ann.py) which processes a single image using the trained model, predicts the road sign, and then triggers motor control logic accordingly.

## Technical Details

- **Neural Network Architecture:**  
  The project uses Convolutional Neural Networks (CNNs) for feature extraction and classification. Architectures include variations of MobileNet and custom CNNs. See [`BotProject.ipynb`](notebooks/BotProject.ipynb) for detailed design iterations and experiments.

- **Data Augmentation:**  
  Data augmentation is used in scripts (like [`trainv5.py`](src/trainv5.py)) with techniques such as shear, zoom, and horizontal flipping to improve generalization.

- **Callbacks and Early Stopping:**  
  Early stopping is employed during training to avoid overfitting by monitoring validation loss.

- **Visualization:**  
  Training progress (accuracy and loss curves) and data distributions are visualized using Matplotlib and Seaborn.

## Future Improvements

- Consolidate training scripts into a single configurable module.
- Update or remove legacy TensorFlow v1 conversion code.
- Enhance data preprocessing routines and add logging.
- Add unit tests to improve code reliability.

## Contact

For further questions or discussions, please contact the project maintainer or open an issue on GitHub.

