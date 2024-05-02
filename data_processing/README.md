# Data Processing

The released ECInstruct dataset contains 92,022 training samples, 9,253 validation samples, 9,253 in-domain test samples and 6,000 out-of-domain test samples. To enable customized size of the ECInstruct dataset, we release all data processing scripts to implement the processing steps detailed in Section A of the paper.
All the scripts are written in jupyter notebook. If you want the ECInstruct dataset with customized size, please follow the instruction to generate processed data.

## Abbreviation of Tasks
| Abbreviation | Full Name |
| --- | --- |
| AVE | Attribute Value Extraction |
| PRP | Product Relation Prediction |
| PM | Product Matching |
| SA | Sentiment Analysis |
| SR | Sequential Recommendation |
| MPC | Multi-class Product Classification |
| PSI | Product Substitute Identification |
| QPR | Query Product Ranking |
| AP | Answerability Prediction |
| AG | Answer Generation |

## Run the Code
To customize the dataset size for each task, please change the following parameters:
- `train_size`: training set size for the task.
- `vt_size`: validation set size and test set size of the task. We only use this parameter in the AVE task and make the same size of validation set and test set in the AVE task.
- `val_size`: validation set size for the task.
- `test_size`: test set size for the task.

### AVE
To process the dataset for the AVE task, please follow the instructions of [MAVE dataset](https://github.com/google-research-datasets/MAVE) and [MAVE code](https://github.com/google-research/google-research/tree/master/mave). 
The default raw data path is `./datasets/mave/datasets/splits/PRODUCT`. Please run the script cell by cell to process and sample the dataset for the AVE task. 
To customize the dataset size, please change the parameters `train_size` and `vt_size`.

### PRP
To process the dataset for the PRP task, please download the product metadata from [Amazon Review 2018](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) and put the metadata in `./datasets/Amazon_review/meta/all_category`. Change the parameters `train_size`, `val_size`, and `test_size` to get the dataset of different size for PRP task.

### PM
To process the dataset for the PM task, please download the raw data from [Amazon-Google Products](https://dbs.uni-leipzig.de/files/datasets/Amazon-GoogleProducts.zip) and put the data in `./datasets/Amazon_Google_products`. Due to the limited raw dataset size, we don't do the sampling in the PM task.

### SA
To process the dataset for the SA task, please download the review data from [Amazon Review 2018](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) and put the review data in `./datasets/Amazon_review/review/all_category`. Change the parameters `train_size`, `val_size`, and `test_size` to get the dataset of different size for SA task.

### SR
To process the dataset for the SR task, please download the product metadata from [Amazon Review 2018](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) and put the metadata in `./datasets/Amazon_review/meta/all_category`, the 2014 version product metadata from [Amazon Review 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) and put the metadata in `./datasets/Amazon_review/meta/all_category_2014`, the review data from [Amazon Review 2018](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) and put the review data in `./datasets/Amazon_review/review/all_category`. Change the parameters `train_size` and `test_size` to get the dataset of different size for SR task.

### MPC, PSI, and QPR
To process the dataset for the MPC, PSI, and QPR task, please download the raw data from [Shopping Queries Dataset](https://github.com/amazon-science/esci-data) and put the data in `./datasets/shopping_queries_dataset`. This script eliminate the non-English data and non-language notations.
To generate the dataset of different size, please change the parameter `train_size`, `val_size`, and `test_size`.

### AP and AG
To process the dataset for the AP and AG task, please download the raw data from [AmazonQA](https://github.com/amazonqa/amazonqa) and put the data in `./datasets/productQA`. Change the parameters `train_size`, `val_size`, and `test_size` to get the dataset of different size for AP or AG task.