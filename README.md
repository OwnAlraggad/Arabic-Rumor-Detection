# Arabic Misinformation(Rumor) Detection

End-to-end NLP pipeline for detecting Arabic misinformation in social media text using the ArCOV19-Rumors dataset (3.5K tweets). The project includes full data preprocessing, feature engineering, and baseline model development for binary rumor classification.

currently:
The system leverages engineered linguistic and contextual features combined with classical machine learning models (XGBoost) to establish strong baseline performance. Evaluation is conducted using multiple metrics including F1-macro, ROC-AUC, and PR-AUC to handle dataset imbalance and ensure robust classification performance.

Key results:

- F1-macro:  0.8947 

- ROC-AUC:   0.9502 

- PR-AUC:    0.9561

Tech stack: Python, Pandas, Scikit-learn, XGBoost, NLP preprocessing, Feature Engineering, Evaluation Metrics

Focus is on building a reproducible baseline pipeline with clear extensibility toward multi-class topic classification and production deployment using FastAPI.

This project was inspired by JOSA’s mission to use open-source AI for social impact. I built it to demonstrate Arabic NLP skills for misinformation detection during the COVID-19 pandemic, using the ArCOV19-Rumors dataset.
