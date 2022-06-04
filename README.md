# nlp_course
nlp course project winter 2021/2022
Question Answering Domain-Generalization With Domain Adversarial Training

## Data Preparation

### Download the original data

- Download the data by running shell file.
- Then run the code. Preprocessed train data will be created before training (It will takes quite a long time)

```bash
$ cd data
$ ./download_data.sh
```

### (Optional) Download the pickled data (for fast data loading)

- Download the pickled data from this [link](https://drive.google.com/open?id=1-IHdLL4oLOI_Ur8ej-KUZ4kVGGuSKcJ2).

- Unzip the zipfile on the root directory.

```bash
.
├── ...
├── pickled_data_bert-base-uncased_False
│   ├── HotpotQA.pkl
│   ├── NaturalQuestions.pkl
│   ├── NewsQA.pkl
│   ├── SQuAD.pkl
│   ├── SearchQA.pkl
│   └── TriviaQA.pkl
└── ...

```

## Requirements

Please install the following library requirements specified in the **requirements.txt** first.

```bash
torch==1.1.0
pytorch-pretrained-bert>=0.6.2
json-lines>=0.5.0
```

## Model Training & Validation
To run our model BertQA with domain-adversarial :
python main.py --model_name bert_da

To run BertQA finetuned on all source domains :
python main.py --model_name bert_all

To run BertQA finetuned on squad :
python main.py --model_name bert_squad


because of the nature of the task (question answering with long paragraph input on bert-base) it will take a lot of time to run the evaluation experiment:
~15m per out-of-domain light dataset (BioASQ, DROP, RelationExtraction)
~1h per out of domain heavy (long context - see paper) dataset (DuoRC,RACE,Textbook)
if you want to check only for the light datasets just remove the rest from the dev file. 
