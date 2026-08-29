\# NeuroBat



\*\*AI-Powered Cricket Batting Biomechanics and Performance Analysis Platform\*\*



NeuroBat is a BSc (Hons) Data Science final-year project that applies Computer Vision, Human Pose Estimation, biomechanical feature engineering and Machine Learning to cricket batting video analysis.



\## Key Features



\- User registration and authentication

\- Player profile management

\- Cricket batting video upload

\- OpenCV frame processing

\- MediaPipe Pose estimation

\- Metric-specific landmark quality assessment

\- Batting phase detection

\- Frame-by-frame biomechanics inspection

\- Biomechanical feature extraction

\- Structured Machine Learning dataset generation

\- XGBoost shot-type classification

\- PostgreSQL analysis-session storage

\- Player analysis history and performance monitoring



\## Batting Phases



NeuroBat analyses five principal phases:



1\. Setup

2\. Backlift

3\. Downswing

4\. Estimated Impact

5\. Follow-through



Estimated Impact is a pose-derived wrist-kinematic proxy and does not represent confirmed bat-ball contact.



\## Machine Learning



The proof-of-concept Machine Learning component uses XGBoost to classify:



\- Cut

\- Drive

\- Pull



The pilot experiment uses structured biomechanical features extracted from quality-approved batting recordings.



The current ML model should be interpreted as an academic proof-of-concept rather than a production-level cricket-shot classifier.



\## Technologies



\- Python

\- Flask

\- OpenCV

\- MediaPipe Pose

\- NumPy

\- Pandas

\- PostgreSQL

\- XGBoost

\- scikit-learn

\- HTML

\- CSS

\- JavaScript



\## Project Scope



NeuroBat is designed as a coaching-support and performance-analysis tool. It is not intended to replace qualified cricket coaches or biomechanics professionals and does not claim laboratory-grade three-dimensional biomechanics from single-camera video.



\## Academic Project



BSc (Hons) Data Science  

Final Year Project — 2026

