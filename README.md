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
- [TensorFlow](https://www.tensorflow.org/) (Keras API used within it)
- OpenCV
- NumPy, Pandas, Matplotlib, and Seaborn
- Other dependencies listed in the requirements file (if available)

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://your-repository-url.git
   cd Road-sign-classification-on-keras
   ```

2. **Create a virtual environment and install dependencies:**

   ```sh
   python -m venv venv
   source venv/bin/activate      # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

   _Note: Create a `requirements.txt` if one is not already provided._

### Folder Organization

- **/src:** Contains all source code for training, testing, and inference.
  - [`chatpi.py`](chatpi.py): Preprocesses images and runs model training and evaluation.
  - [`ann.py`](ann.py): Serves as an entry point for model inference.
  - [`convert.py`](convert.py): Responsible for conversion tasks of models (if applicable).
  - [`modelchange.py`](modelchange.py): Contains legacy TensorFlow v1 code for model conversion.
  - Training scripts (`Training.py`, `traintrain.py`, `trainv5.py`): Explore different training configurations and architectures.
- **/models:** Stores all models and weights generated throughout the project.
- **/data:** Holds CSV files and any additional data required for model training or evaluation.
- **/notebooks:** Contains Jupyter notebooks (e.g., [`BotProject.ipynb`](BotProject.ipynb)) used for prototyping, visualization, and experiments.

## Usage

### Data Preparation

Place your training and testing data (images and labels) in the appropriate locations. The project uses CSV files and a folder of images to create datasets. Check the data loading sections in [`chatpi.py`](chatpi.py) for guidance on file structure.

### Training

Adjust parameters such as image size, batch size, and epochs in the training scripts in `/src`. For instance, [`chatpi.py`](chatpi.py) uses a batch size of 64 and resizes images to 128×128 for training.

Run the training script:

```sh
python src/chatpi.py
```

Alternatively, use the provided Jupyter Notebook in `/notebooks` for an interactive experience:

```sh
jupyter notebook notebooks/BotProject.ipynb
```

### Evaluation

After training, the models can be evaluated using the test datasets. The scripts print metrics including accuracy and loss after evaluation. Look inside the training scripts and notebook cells for evaluation commands.

### Model Conversion

For model conversion, such as converting to TensorFlow Lite:
- Use [`trainv5.py`](trainv5.py) for training and conversion to `.h5` or `.tflite` as required.
- Legacy conversion code exists in [`modelchange.py`](modelchange.py) for frozen graph conversion (TensorFlow v1). Remove this if you rely solely on TensorFlow v2 workflows.

### Inference

For inference, use [`ann.py`](ann.py) which processes a single image using the trained model, predicts the road sign, and then triggers motor control logic accordingly.

## Technical Details

- **Neural Network Architecture:**  
  The project uses Convolutional Neural Networks (CNNs) for feature extraction and classification. Architectures explored include variations of MobileNet and custom CNNs. See [`BotProject.ipynb`](notebooks/BotProject.ipynb) for design iterations and experiments.

- **Data Augmentation:**  
  Data augmentation is applied in scripts such as [`trainv5.py`](trainv5.py) to improve model generalization with techniques including shear, zoom, and horizontal flipping.

- **Callbacks and Early Stopping:**  
  During training, callbacks like early stopping are used to avoid overfitting by monitoring validation loss.

- **Visualization:**  
  Matplotlib and Seaborn are used to visualize training progress (accuracy, loss curves) and data distributions.

## Future Improvements

- Consolidate training scripts into a single configurable module.
- Update legacy TensorFlow v1 conversion code if moving entirely to TensorFlow v2.
- Enhance data preprocessing routines and add logging.
- Add unit tests to improve code reliability.

## Contact

For further questions or discussions, please contact the project maintainer or open an issue on GitHub.


