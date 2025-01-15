# Speech-Emotion-Recognition-using-Machine-Learning

## Project Overview

This project bridges the gap between human communication and artificial intelligence by implementing a **Speech Emotion Recognition (SER)** system. It focuses on classifying emotions such as `angry`, `sad`, `happy`, `neutral`, `fearful`, `disgust`, and `surprised` using **classical machine learning algorithms** like Random Forests and Support Vector Machines (SVMs). The project leverages **acoustic features** and integrates diverse datasets to build a robust model.

## Key Features

- **Datasets Used**:
  - **RAVDESS**: High-quality recordings from 24 actors.
  - **TESS**: Canadian English emotional speech dataset.
  - **SAVEE**: British English dataset emphasizing emotional clarity.
  - **CREMA-D**: Diverse accents and vocal styles with over 7,000 audio files.

- **Audio Features Extracted**:
  - **MFCCs**, **jitter**, **shimmer**, **spectral contrast**, **chroma features**, and **pitch-related features**.
  - Temporal and spectral features highlighting voice intensity and emotional nuances.

- **Preprocessing Techniques**:
  - Data augmentation: **Pitch shifting**, **noise addition**, and **time stretching**.
  - Class balancing with **SMOTE**.
  - Feature standardization.

- **Machine Learning Models**:
  - Random Forests and SVMs were trained and fine-tuned using cross-validation.

- **Performance Evaluation**:
  - Accuracy, precision, recall, and F1-scores were calculated for all models.
  - Detailed **error analysis** with PCA and clustering visualizations.

## Key Code Files

- **`audio_feature_extraction.ipynb`**: Visualizing features for a single dataset.
  - Audio data extracted from a single file for visualization.

- **`creating_dataset.ipynb`**: Preprocessing and dataset creation pipeline.
  - Combines RAVDESS, TESS, SAVEE, and CREMA-D datasets.
  - Implements feature extraction using the `Librosa` library.
  - Applies data augmentation techniques like pitch shifting and noise addition.

- **`data_exploration.ipynb`**: Dataset visualization and feature analysis.
  - Generates correlation matrices and boxplots for feature analysis.
  - Performs PCA to understand feature separability across emotions.

- **`combined_random_forest.ipynb`**: Training and evaluation of Random Forest models.
  - Implements GridSearchCV for hyperparameter tuning.
  - Includes confusion matrices and performance metrics for individual and combined datasets.

- **`combined_svm.ipynb`**: Training and evaluation of SVM models.
  - Uses stratified cross-validation for balanced class representation.
  - Presents confusion matrices and comparative analysis with Random Forest.

- **`Evaluation_of_[Dataset].ipynb`**: Per-dataset evaluations (e.g., RAVDESS, TESS, SAVEE, CREMA-D).
  - Tests Random Forest and SVM on each dataset individually.
  - Analyzes strengths and weaknesses for specific datasets.

## Results

1. **Individual Datasets**:
   - Achieved high accuracy on RAVDESS and TESS using both Random Forests and SVMs.
   - Lower performance on CREMA-D due to high variability in accents and recording conditions.

2. **Combined Datasets**:
   - **Random Forests**: 72% weighted accuracy, performing well for `angry` and `surprised`.
   - **SVMs**: 70% weighted accuracy, excelling in high-dimensional separation but struggling with overlapping emotions.

3. **After Data Augmentation**:
   - Slight performance improvement for underrepresented classes like `fearful`.

## Project Structure

- **Notebooks**:
  - `audio_feature_extraction.ipynb`: Features visualization.
  - `creating_dataset.ipynb`: Preprocessing and dataset creation.
  - `data_exploration.ipynb`: Feature analysis and insights.
  - `combined_random_forest.ipynb`: Random Forest training and evaluation.
  - `combined_svm.ipynb`: SVM training and evaluation.
  - `Evaluation_of_[Dataset].ipynb`: Individual dataset evaluations.

- **Report**:
  - `ML_Project_Report.pdf`: Comprehensive project documentation and results.

## How to Use

1. Clone this repository.
   ```bash
   git clone https://github.com/your-username/emotion-detection-speech.git
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks in the order listed under Project Structure.

## Contributors

- **Hamza Riaz**  
  NUST, Islamabad, Pakistan
  [hriaz.bscs22seecs@seecs.edu.pk](mailto:hriaz.bscs22seecs@seecs.edu.pk)

- **Maha Baig**  
  NUST, Islamabad, Pakistan  
  [mbaig.bscs22seecs@seecs.edu.pk](mailto:mbaig.bscs22seecs@seecs.edu.pk)
