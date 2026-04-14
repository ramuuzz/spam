# SMS Spam Classifier Web App

This project is an end-to-end Machine Learning pipeline and web application designed to classify text messages as either **Spam** or **Not Spam (Ham)**. It pairs a machine learning model developed with Python and Scikit-Learn with a frontend user interface powered by Django.

## Project Structure

The project is broadly divided into two main components:

1. **Model Discovery & Training**
   - `spam.csv`: The dataset containing raw SMS messages.
   - `spam.ipynb`: A Jupyter Notebook used to load the dataset, perform Exploratory Data Analysis (EDA), preprocess text (tokenization, stop words removal, stemming using `NLTK`), train, and export the classification model.
   
   **Model Performance Metrics (MultinomialNB with TF-IDF):**
   - **Accuracy**: 97.10%
   - **Precision**: 100%
   - **Confusion Matrix**:
     ```text
     [[896   0]
      [ 30 108]]
     ```
   
2. **Django Web Application** (`spam_class`)
   - `manage.py`: Django's command-line utility for administrative tasks.
   - `spamm/`: The main Django app containing the views and prediction logic.
   - `spamm/views.py`: This receives text input, preprocesses it to match the training pipeline, vectorizes it using TF-IDF (`vectorizer.pkl`), and makes live predictions using the pre-trained ML model (`model.pkl`).
   - `spamm/templates/index.html`: The user interface where users can paste their messages and see the predicted outcome.

## Prerequisites

Before running the application, make sure you have the required packages installed. It's recommended to use a virtual environment (`.venv`).

```bash
pip install pandas numpy matplotlib scikit-learn nltk django
```

Additionally, you may need to download the required NLTK corpora. This is automatically handled in the provided notebook, but manually verify that the `punkt` and `stopwords` data sets are available for NLTK if you run into any issues.

## How to Run the Django Web App

1. Open your terminal or powershell in the root directory of the `spam` project.
2. Activate your virtual environment (if you are using one).
3. Change directories into the `spam_class` folder where the Django application resides:
   ```bash
   cd spam_class
   ```
4. Run the local development server:
   ```bash
   python manage.py runserver
   ```
5. Open a web browser and navigate to the address shown in your terminal (typically `http://127.0.0.1:8000/`).

## How to Work With the Model

To view or retrain the Data Science portion:
1. Open up Jupyter Notebooks or your preferred notebook editor.
2. Open `spam.ipynb`.
3. Re-run or modify the cells to adjust text preprocessing steps, change out the machine learning algorithm, and serialize newer versions of `model.pkl` and `vectorizer.pkl`. 

Whenever you retrain, you will need to replace the `.pkl` files within the `spam_class/spamm` directory to update the live web application.
