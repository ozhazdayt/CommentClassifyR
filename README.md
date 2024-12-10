CommentClassifyR

CommentClassifyR combines AI and RPA to automate the categorization of product reviews. Using Python and UiPath, this project collects, categorizes, and reports review data in an integrated and efficient workflow.
________________________________________
Project Structure
CommentClassifyR/

├── Python_Part/       # Contains scripts and datasets for data processing and ML

│   ├── model.py       # Main script for machine learning

│   ├── GUI.py         # GUI-based processing (optional)

│   ├── test.py        # Testing script

│   ├── Dataset.xlsx   # Dataset used for training/testing

│   ├── amazon_reviews.xlsx           # Input dataset

│   └── amazon_reviews_categorized.xlsx  # Categorized output

├── UiPath_Part/       # Contains UiPath workflows and configurations

│   ├── Main.xaml      # Main UiPath workflow

│   ├── Data/Config.xlsx  # Configuration file

│   ├── Data/Input/    # Input files directory

│   ├── Data/Output/   # Output files directory

│   └── Framework/     # Supporting workflows for UiPath

________________________________________

Prerequisites

Software Requirements

•	Python 3.x

•	UiPath Studio

Libraries for Python

Ensure you have the following libraries installed:

•	pandas

•	openpyxl

•	nltk

You can install them via:

pip install pandas openpyxl nltk

________________________________________

Setup

UiPath Setup

1.	Install UiPath Studio and required extensions.
3.	Open UiPath_Part/Main.xaml.
4.	Update paths in UiPath_Part/Data/Config.xlsx to match your local setup.
Python Setup
1.	Update paths in Python_Part/model.py as necessary.
2.	Ensure all required libraries are installed.
________________________________________
How to Run
1.	Update Configuration Files
o	Edit Config.xlsx (UiPath) and model.py (Python) to set the correct file paths.
2.	Run the Workflow
o	Open UiPath Studio and execute the Main.xaml file.
3.	Automated Processing
o	UiPath will invoke the Python script for machine learning, categorize reviews, and export results to amazon_reviews_categorized.xlsx.
o	The final report is emailed to stakeholders as per the UiPath workflow.
________________________________________
Notes
•	Make sure to place input files in UiPath_Part/Data/Input/.
•	Output files will be saved in UiPath_Part/Data/Output/.
•	Python and UiPath paths must align for the integration to work.

