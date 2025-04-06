# Airbnb-Data-Analysis-Project

### Objective and Scope
#### Objective: The goal of this project is to analyze Airbnb listing data to uncover meaningful insights related to pricing, availability patterns, sentiment trends, and clustering for segmentation. Additionally, the project provides actionable recommendations to optimize Airbnb listings and improve decision-making.

#### Specific Objectives:

Understand the relationship between numeric features and target variables like price and availability.

Develop and evaluate machine learning models for prediction and classification.

Utilize clustering for segmentation of listings.

Provide actionable e-commerce strategies and recommendations based on data insights.

#### Scope:

Exploratory Data Analysis (EDA) to understand data distribution, detect anomalies, and explore feature relationships.

Preprocessing steps like handling missing values, encoding categorical variables, and scaling numerical features.

Development of machine learning models for regression, classification, and clustering.

Evaluation of model performance using appropriate metrics and optimization techniques.

Actionable insights generation and visualization.

## Methodology
Data Preprocessing:

Data Cleaning: Handled missing and infinite values with imputation using column means.

Feature Engineering: Created derived features like weighted_rating, availability_proportion, and review_frequency.

Scaling: Used StandardScaler to ensure consistent feature distribution across models.

## Model Development:

Regression Analysis:

Used Random Forest Regressor for predicting price.

Metrics evaluated: RMSE, R², and MAE.

Classification:

Categorized availability into Low, Medium, and High using thresholds.

Used Random Forest Classifier for accurate prediction.

Clustering:

Applied KMeans for segmenting listings into clusters like budget-friendly, premium, and highly available.

Used Elbow Method and Silhouette Scores for optimization.

NLP and Sentiment Analysis:

Text Preprocessing: Cleaned review data by removing punctuation, numbers, and stop words.

Feature Conversion: Used TF-IDF to transform reviews into numerical features for sentiment classification.

Sentiment Prediction: Classified reviews into Positive, Neutral, or Negative sentiments using Random Forest.

### Results and Insights
#### Price Prediction Model Report
Model: Random Forest Regressor Objective: Predict the price of listings based on features like reviews_per_month, availability_proportion, and weighted_rating.

Performance Metrics:

    1.Mean Squared Error  (MSE): 0.00

    2.R-squared (R²): 1.00

    3.Root Mean Squared Error (RMSE): 0.04

    4.Mean Absolute Error (MAE): 0.01

Insights:

Exceptional accuracy in price prediction.

Key features influencing predictions included review frequency, adjusted ratings, and availability metrics.

Achieved high prediction accuracy for price.

Revealed key features like reviews_per_month, availability_proportion, and adjusted_rating.

#### Classification:Availability Prediction Model Report
Model: Random Forest Classifier Objective: Classify listings into availability categories (Low, Medium, High) based on features like price, reviews_per_month, and weighted_rating.

Performance Metrics:

    1.Accuracy: 1.00

2.Classification Report:

    Category     Precision	 Recall	     F1-Score	  Support

    Low	        1.00	    1.00	   1.00	     2004

Successfully categorized availability into Low, Medium, and High.

Enhanced understanding of accessibility patterns.
### Insights:

Perfect classification performance.

Helps identify accessibility trends and informs booking strategies for listings with varying availability levels.

#### Sentiment Analysis:Sentiment Classification Model Report
Model: Random Forest Classifier Objective: Classify customer reviews into sentiments (Positive, Neutral, Negative) using TF-IDF features derived from review text.

Performance Metrics:

    1.Sentiment Analysis Accuracy: 1.00

2.Classification Report:

    Sentiment	Precision  Recall	 F1-Score	Support

    Neutral	        1.00	   1.00	        1.00	   386

    Positive	  1.00	      1.00	     1.00	     1618

Sentiment classification provided insights into customer feedback trends, identifying satisfaction drivers and areas for improvement.
Insights:

Perfect sentiment prediction, providing valuable insights into customer satisfaction and areas for improvement.

#### Clustering:Clustering Analysis Report
Model: KMeans Objective: Segment listings into clusters based on features like price, availability_365, reviews_per_month, and number_of_reviews.

Performance Metrics:

    k	  Silhouette Score

    2	    0.8089

    3	    0.6149

    4	    0.4830

    5	    0.4838

    6	    0.3089

    Best k: 2 (Silhouette Score: 0.8089)

Insights:

Identified two distinct clusters with strong separation:

Cluster 1: Budget-friendly listings with lower prices and fewer reviews.

Cluster 2: High-value listings with premium prices and strong review scores.

Optimal clusters revealed patterns across price, reviews, and availability.

Segmentation enabled targeted marketing and enhanced strategic decisions.

#### Actionable Recommendations
Dynamic Pricing:

Adjust pricing based on demand trends and seasonal availability patterns.

Promote value-based pricing for long-term bookings in high-availability clusters.

Targeted Marketing:

Tailor campaigns based on cluster segmentation (e.g., premium vs. budget listings).

Highlight positive sentiment trends in listing descriptions to attract bookings.

Customer Experience:

Address common concerns highlighted in negative reviews to improve service quality.

Leverage sentiment analysis to identify and enhance guest satisfaction drivers.

Dashboard Implementation:

Create interactive dashboards for hosts to visualize insights dynamically.

Incorporate clustering, pricing predictions, and sentiment trends for easy exploration.

#### Limitations
Imputation assumptions for missing values might impact prediction accuracy.

Clustering based solely on numeric features excludes categorical insights like property type or location.

Sentiment analysis accuracy depends heavily on the quality of review text.

Potential for Further Exploration
Incorporate deep learning models for sentiment classification to handle nuanced language patterns.

Explore geographical segmentation to uncover regional trends in pricing and availability.

Enhance clustering by including categorical features and behavior-driven metrics.

Optimize models further using ensemble methods or hyperparameter tuning.

Develop real-time predictive dashboards for pricing and availability management.

### Environment Setup
Prerequisites:

Programming Language: Python 3.12

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, NLP tools like NLTK or TextBlob.

Tools: Jupyter Notebook or other Python-compatible IDEs.

#### Setup Instructions:

Install required libraries using pip install.

Run the notebooks sequentially to replicate preprocessing, modeling, and visualization.
