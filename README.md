# Sub-event Detection in Twitter streams

The goal of this project is to study and apply machine learning/artificial intelligence techniques to predict the presence of specific sub-events in tweets posted during football games from the 2010 and 2014 World Cups.

## Setup

To clone the project and install the dependencies, run the following commands:

```bash
git clone https://github.com/l4cer/twitter-sub-event-detection.git
cd twitter-sub-event-detection
python -m venv venv
venv/Scripts/activate     # Windows
source venv/bin/activate  # Linux
pip install -r requirements.txt
```

## Dataset

Download the dataset from the following [link](https://www.kaggle.com/competitions/sub-event-detection-in-twitter-streams/data) and place it inside the project folder following this structure:

```
📁 twitter-sub-event-detection
   📁 eval_tweets
   📁 train_tweets
   📄 .gitignore
   📄 console.py
   📄 evaluate.py
   📄 models.py
   📄 preprocessing.py
   📄 README.md
   📄 requirements.txt
```

## Execution

The code will create two folders inside the project folder:

```
📁 twitter-sub-event-detection
   📁 data
   📁 predictions
   ...
```

These folders will store the preprocessed data and the CSV files with predictions properly formatted for [Kaggle](https://www.kaggle.com/competitions/sub-event-detection-in-twitter-streams/overview).

To preprocess the dataset (only needs to be run once at the start), execute the command:

```bash
python preprocessing.py
```

⚠️ **Note:** this command will take about 5 to 7 minutes to complete, but you can monitor the progress through the console/terminal.

Create your model in the `models.py` file and call it in `evaluate.py`. To run the code and generate the final CSV, execute the command:

```bash
python evaluate.py
```

The CSV file with the chosen model's name should have been created inside the `predictions` folder. Now, simply upload it to Kaggle to evaluate your model's performance.
