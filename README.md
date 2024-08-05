Clustering of Learnings

The project is for clustering learnings based on semantic similarities using Natural Language Processing. The input data is in JSONL format, and the project pulls on machine learning models for extracting features from text and grouping similar learnings.

Features
  Data Loading: Supports data loading from JSONL files.
  Text Preprocessing: Numerical features represent the textual data through the TF-IDF vectorization process.
  It applies KMeans clustering to cluster similar learnings. 
  Visualization: After that, it visualizes the result in clusters using PCA for dimensional reduction. Get Started

Prerequisites
  Python 3.7+
  Required Python libraries were install via requirements.txt in VSC

Current Status
In the development of this project, I have faced some heavy issues with this dataset. These are formatting mistakes in the JSONL file. These defects caused failure in parsing and processing data successfully.

Impact on the Project
Due to these issues, I have not been able to run the Clustering Algorithms, and illustrate that clustering has taken place. The scripts are runable, but they need a correctly formatted dataset in order to test them.

Next Steps
I tried working on fixing up the dataset issues; however, no revelent progress seems to have taken place if any of you have experience with JSONL data or would like to contribute, help would be most welcome on this end.
