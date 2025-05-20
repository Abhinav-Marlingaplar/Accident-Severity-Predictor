# Accident Severity Prediction using Random Forest Classifier

## Overview

This project aims to predict the severity of road traffic accidents using a Machine Learning approach. By analyzing various contributing factors, the model can classify accident severity levels, providing valuable insights for proactive road safety measures, optimizing emergency response, and resource allocation. The core of this solution is a **Random Forest Classifier**, chosen for its robustness, high accuracy, and ability to handle diverse datasets. A Streamlit web application is provided to demonstrate the model's predictive capabilities interactively.

## Table of Contents

- [Project Description](#project-description)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model & Methodology](#model--methodology)
- [Evaluation Results](#evaluation-results)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Live Demo (Optional - Add Link Here)](#live-demo-optional---add-link-here)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Description

Road traffic accidents are a significant global concern, leading to severe human and economic consequences. Predicting the severity of these incidents can play a crucial role in mitigating their impact. This project addresses the challenge by developing a machine learning model that classifies accident severity based on factors such as environmental conditions, road characteristics, and other relevant features. The ultimate goal is to empower emergency services, policymakers, and urban planners with predictive insights to enhance road safety and optimize response strategies.

## Key Features

-   **Accident Severity Prediction:** Utilizes a trained Random Forest Classifier to predict the severity level of an accident.
-   **Interactive Streamlit Application:** Provides a user-friendly web interface for inputting accident parameters and getting real-time severity predictions.
-   **Model Persistence:** The trained model is saved for easy loading and deployment, eliminating the need for retraining.
-   **Hyperparameter Optimization:** Employs `RandomizedSearchCV` for efficient and effective tuning of the Random Forest model's hyperparameters.

## Dataset

The project utilizes the `Accident-Data.csv` dataset, which contains various features related to road accidents. Due to its size (~188 MB), the dataset is managed using Git Large File Storage (Git LFS) in this repository.

-   **File Name:** `Accident-Data.csv`
-   **Size:** Approximately 188 MB
-   **Contents:** Includes columns related to accident time, location, weather, road conditions, vehicle types, and the target variable (accident severity).

## Model & Methodology

The core predictive model is a **Random Forest Classifier**.

The machine learning pipeline involves the following key steps:
1.  **Data Loading & Preprocessing:** Handling missing values, encoding categorical features, and scaling numerical features.
2.  **Exploratory Data Analysis (EDA):** Understanding feature distributions and relationships with accident severity. (Details in `Accident-Severity-Predictor.ipynb`)
3.  **Model Training:** Training a `RandomForestClassifier` on the preprocessed data.
4.  **Hyperparameter Tuning:** Efficiently optimizing the model's hyperparameters (e.g., `n_estimators`, `max_depth`, `min_samples_split`, `max_features`) using `sklearn.model_selection.RandomizedSearchCV` with 5-fold cross-validation. This ensures a robust and high-performing model.
5.  **Model Persistence:** The best performing model (`best_rf.joblib`) and the list of features used (`feature_columns.json`) are saved for direct use in the Streamlit application.

## Evaluation Results

The Random Forest Classifier demonstrated strong predictive capabilities on the unseen validation set.

**Validation Set Performance:**

| Severity Class | Precision | Recall | F1-Score | Support |
| :------------- | :-------- | :----- | :------- | :------ |
| 1              | 0.91      | 0.92   | 0.91     | 855     |
| 2              | 0.67      | 0.55   | 0.61     | 855     |
| 3              | 0.71      | 0.71   | 0.71     | 855     |
| 4              | 0.69      | 0.81   | 0.74     | 855     |

**Overall Accuracy:** `0.747`

The model exhibits high accuracy, particularly for the most frequent severity class (Class 1). While there's room for improvement in predicting less frequent classes (2, 3, 4), the overall performance provides a solid foundation for practical application.

## Project Structure
Accident_Severity_Predictor/
├── model/
│   ├── best_rf.joblib             # Trained Random Forest model
│   └── feature_columns.json       # List of feature columns used by the model
├── Accident-Data.csv              # The dataset used for training (LFS-tracked)
├── Accident-Severity-Predictor.ipynb # Jupyter Notebook with EDA, training, and evaluation
├── app.py                         # Streamlit web application
└── requirements.txt               # Python dependencies

## How to Run Locally

To set up and run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Abhinav-Marlingaplar/Accident-Severity-Predictor.git](https://github.com/Abhinav-Marlingaplar/Accident-Severity-Predictor.git)
    cd Accident_Severity_Predictor
    ```
    *Note: Ensure Git LFS is installed (`git lfs install`) before cloning to correctly download `Accident-Data.csv`.*

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    This command will open the Streamlit application in your web browser, typically at `http://localhost:8501`.

## Live Demo (Optional - Add Link Here)

*(Once you deploy your Streamlit app to Streamlit Community Cloud or another platform, you can add a link here like this:)*

Experience the live application here: [https://your-streamlit-app-url.streamlit.app/](https://your-streamlit-app-url.streamlit.app/)

## Future Enhancements

-   **Advanced Feature Engineering:** Explore more complex feature interactions and derivations from the existing dataset.
-   **Handling Class Imbalance:** Implement techniques like SMOTE, ADASYN, or use custom loss functions to improve prediction for minority severity classes.
-   **Model Interpretability:** Employ tools like SHAP or LIME to explain individual predictions and gain deeper insights into feature importance.
-   **Real-time Data Integration:** Explore possibilities of integrating with real-time traffic or weather data sources.
-   **Deployment Optimization:** Further optimize the Streamlit application for faster loading and scalability.

## Contributing

Contributions are welcome! If you have suggestions for improvements, feature additions, or bug fixes, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you create one).

## Contact

For any questions or inquiries, please feel free to reach out:

-   **Abhinav Marlingaplar**
-   GitHub: [Abhinav-Marlingaplar](https://github.com/Abhinav-Marlingaplar)
