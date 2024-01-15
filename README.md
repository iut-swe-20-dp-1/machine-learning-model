# Stress Prediction Machine Learning Project

## Overview
This repository contains the code for a machine learning project that aims to generate a stress score and a stress class using 3 physiological data readings (HR, TEMP, EDA). The project utilizes pandas, scikit-learn and 2 RandomForest models (one Classifier and one Regressor) for building and training machine learning models.

## Table of Contents
- [Environment](#environment)
- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Training](#training)
- [Results](#results)
- [License](#license)
- [Contributors](#contributors)

## Environment
   - Ensure you have a Python 3.x environment set up on your machine.
   - Make sure to download the dataset and setup the folder structure provided above.

## Installation
To set up the project, follow these steps:

1. Clone the repository
2. Download the AffectiveROAD dataset from [here](https://dam-prod2.media.mit.edu/x/2021/06/14/AffectiveROAD_Data_w1dqSB9.zip)
3. Copy the AffectiveROAD_Data folder inside the unzipped folder and paste it into the folder where the repository was clonned.
4. First run the model_training.ipynb file.
6. Collect some physiological data and move them into the Model_Training_Data folder.
5. Then run the model_retraining.ipynb file.

## Dataset
The AffectiveROAD Dataset is a comprehensive dataset designed for research in affective computing and human-centered applications. It captures physiological signals, environmental data, and user annotations to provide rich insights into the affective states of individuals in various real-world scenarios.

We used the following physiological data from the dataset:
1. HR (Heart Rate)
2. TEMP (Skin Temperature)
3. EDA (Skin Conductance)

## Models
1. A RandomForestRegressor model was used to generate the stress scores.
2. A RandomForestClassifier model was used to generate the stress classes [0 = "low", 1 = "medium", 2 = "high"].

## Training
We trained the model first on the AffectiveRoadDataset and then further trained the model on the collected data.

## Results
R-score for Regressor = around 82%
Accuracy for Classifier = around 97%

## License

This project is licensed under the [MIT License](LICENSE).

## Contributors

- **Shanta Maria**
  - *GitHub:* [NafisaMaliyat-iut](https://github.com/NafisaMaliyat-iut)

- **Nafisa Maliyat**
  - *GitHub:* [maria-iut1234](https://github.com/maria-iut1234)

- **Ayesha Afroza Mohsin**
  - *GitHub:* [AyeshaMohsin](https://github.com/AyeshaMohsin)