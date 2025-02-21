# ComplaintsAnalysisPortfolio
Introduction: The project uses NLP to analyze consumer complaints and detect major issues which appear within them via applying vectorization and topic modelling techniques.

Required libraries: pandas, numpy, nltk, scikit-learn matplotlib.

File Structure:

src/: contains souce code for the analysis
  o	main.py: Runs the full analysis
  o	cleaning_preprocess.py: Handles text preprocessing
  o	vectorizing.py: Converts text into numerical vectors
  o	topic_extract.py: Applies topic modeling (LDA, NMF)
  o	visuals.py: Generates word frequency and topic coherence plots
  
•	data/: Stores dataset files
  o	(1000)Consumer_Finance_Complaints.csv: minimized data set
  
How to Use: after downloading necessary libraries, modify the "frame" string in main page with the correct file path to enable program access and analysis of the database.

Output: program generates extracted topics, displaying coherence scores and visual representations of word distributions together with topic coherence.

