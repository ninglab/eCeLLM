import fire
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json
import pdb
import re
from datasets import load_dataset

def main(
    label_path: str = "test.json",
    task: str = "",
    setting: str = "",
):
    ## load from google drive
    # PATH = f'ECInstruct/{task}/{setting}'
    # label_list = json.load(open(f'{PATH}/{label_path}', 'r'))
    # for i in range(len(label_list)):
    #     label_list[i] = label_list[i]['output']

    # load from huggingface
    dataset = load_dataset("NingLab/ECInstruct")["train"]
    label_list = []
    for data in dataset:
        if data["split"] == "test" and data["task"] == task and data["setting"] == setting:
            label_list.append(data["output"])

    prediction_path = f'evaluation/{task}/{setting}.json'
    prediction_list = json.load(open(prediction_path, 'r'))
    
    #evaluation for extraction tasks
    if (task == 'Attribute_Value_Extraction'):

        def process_pred(preds):
            preds = preds.replace('[', '').replace(']', '')
            extracted = []
            for pred in preds.split(', \"attribute\"'):
                if not pred.startswith('\"attribute\"'):
                    pred = '\"attribute\"' + pred
                pred = '{'+pred+'}'
                try:
                    res = json.loads(pred)
                except:
                    res = ''
            
            if res != '':
                extracted.append(res)
            return extracted

        idx2pred = {}
        pred_skipped = 0
        avg_pred = 0
        for i in range(len(prediction_list)):

            idx2pred[i] = process_pred(prediction_list[i])
            if len(idx2pred[i]) == 0:
                pred_skipped += 1
                continue
            else:
                avg_pred += len(idx2pred[i])

        avg_pred /= len(idx2pred)
        attribute2count_label = {}
        attribute2count_pred = {}
        attribute2hit = {}
        label_skipped = 0

        for i in range(len(label_list)):

            try:
                labels = json.loads(label_list[i])
            except:
                label_skipped += 1
                continue

            for label in labels:

                if label["attribute"] not in attribute2count_label:
                    attribute2count_label[label["attribute"]] = 0
                attribute2count_label[label["attribute"]] += 1

                for pred in idx2pred[i]:
                    if (pred["attribute"] == label["attribute"] and pred["value"] == label["value"] and pred["source"] == label["source"]):
                            if pred["attribute"] not in attribute2hit:
                                attribute2hit[pred["attribute"]] = 0
                            attribute2hit[pred["attribute"]] += 1

                    elif ('OOD' in setting) and (pred["attribute"] == label["attribute"]) and (pred["value"] != None) and (pred["source"] != None)\
                    and ((str(pred["value"]) in str(label["value"])) or (str(label["value"]) in str(pred["value"]))) and ((pred["source"] in label["source"]) or (label["source"] in pred["source"])):
                            if pred["attribute"] not in attribute2hit:
                                attribute2hit[pred["attribute"]] = 0
                            attribute2hit[pred["attribute"]] += 1

        for idx in idx2pred:
            for pred in idx2pred[idx]:
                
                if ("attribute" in pred):
                    if pred["attribute"] not in attribute2count_pred:
                        attribute2count_pred[pred["attribute"]] = 0
                    attribute2count_pred[pred["attribute"]] += 1
        
        sorted_list = [(k, v) for k, v in sorted(attribute2count_label.items(), key=lambda item: item[1])]
        tops = sorted_list[-5:]
        tops.reverse()

        for i, (k, v) in enumerate(tops):
            if k not in attribute2hit:
                attribute2hit[k] = 0
            if k not in attribute2count_pred:
                attribute2count_pred[k] = 0
                
            recall = attribute2hit[k] / v
            precision = attribute2hit[k] / (attribute2count_pred[k] + 1e-6)
            f1 = (2 * recall * precision) / (recall + precision + 1e-6)

        total_hit = 0
        total_count_label = 0
        total_count_pred = 0
        for k in attribute2hit:
            total_hit += attribute2hit[k]
        
        for k in attribute2count_label:
            total_count_label += attribute2count_label[k]
        
        for k in attribute2count_pred:
            total_count_pred += attribute2count_pred[k]

        total_recall = total_hit / total_count_label
        total_precision = total_hit / (total_count_pred + 1e-6)
        total_f1 = (2 * total_recall * total_precision) / (total_recall + total_precision + 1e-6) 
        print('{"recall": %.3f, "precision": %.3f, "f1": %.3f, "#invalid": %d}' % (total_recall, total_precision, total_f1, pred_skipped))
    
    #evaluation for binary classification tasks
    elif (task == 'Answerability_Prediction') \
        or (task == 'Product_Substitue_Identification') \
        or (task == 'Product_Matching'):
        skipped = 0
        filtered_prediction_list = []
        filtered_label_list = []
        valid_response = ['yes', 'no']
        valid_response = set(valid_response)

        for i in range(len(prediction_list)):
            pred = prediction_list[i].lower().strip()
            if pred in valid_response:
                filtered_prediction_list.append(pred)
                filtered_label_list.append(label_list[i])
            else:
                skipped += 1

        acc = accuracy_score(filtered_label_list, filtered_prediction_list)
        pre_yes = precision_score(filtered_label_list, filtered_prediction_list, pos_label = 'yes', average = 'binary')
        rec_yes = recall_score(filtered_label_list, filtered_prediction_list, pos_label = 'yes', average = 'binary')
        f1_yes  = f1_score(filtered_label_list, filtered_prediction_list, pos_label = 'yes', average = 'binary')

        tn, fp, fn, tp = confusion_matrix(filtered_label_list, filtered_prediction_list).ravel()
        specificity = tn / (tn + fp + 1e-8)
        NPV = tn / (tn + fn + 1e-8)

        print('{"acc": %.3f, "recall": %.3f, "precision": %.3f, "f1": %.3f, "specificity": %.3f, "npr": %.3f, "#invalid": %d}' % (acc, rec_yes, pre_yes, f1_yes, specificity, NPV, skipped))
    
    #evaluation for recommendation tasks
    elif (task == 'Sequential_Recommendation'):
        skipped = 0
        filtered_prediction_list = []
        filtered_label_list = []
        valid_response = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
        valid_response = set(valid_response)

        for i in range(len(prediction_list)):
            pred = prediction_list[i].strip()
            if pred in valid_response:
                filtered_prediction_list.append(pred)
                filtered_label_list.append(label_list[i])
            else:
                skipped += 1
        
        acc = accuracy_score(filtered_label_list, filtered_prediction_list)
        print('{"recall@1": %.3f, "#invalid": %d}' % (acc, skipped))
    
    #evaluation for multi-class classification tasks
    elif (task == 'Multiclass_Product_Classification') \
        or (task == 'Sentiment_Analysis') \
        or (task == 'Product_Relation_Predicition'):

        skipped = 0
        filtered_prediction_list = []
        filtered_label_list = []
        valid_response = ['A', 'B', 'C', 'D', 'E']
        valid_response = set(valid_response)

        for i in range(len(prediction_list)):
            pred = prediction_list[i].strip()
            if pred[0] in valid_response:
                filtered_prediction_list.append(pred[0])
                filtered_label_list.append(label_list[i][0])
            else:
                skipped += 1

        acc = accuracy_score(filtered_label_list, filtered_prediction_list)
        pre_macro = precision_score(filtered_label_list, filtered_prediction_list, average = 'macro')
        rec_macro = recall_score(filtered_label_list, filtered_prediction_list, average = 'macro')
        f1_macro  = f1_score(filtered_label_list, filtered_prediction_list, average = 'macro')

        print('{"acc": %.3f, "recall": %.3f, "precision": %.3f, "f1": %.3f, "#invalid": %d}' % (acc, rec_macro, pre_macro, f1_macro, skipped))
    
    #evaluation for generation tasks
    elif (task == 'Answer_Generation'):
    
        import evaluate
        import torch
        import numpy as np
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        num_skipped = 0
        bertscore = evaluate.load("bertscore")
        bert_score   = bertscore.compute(predictions=prediction_list, references=label_list, lang="en")
        bert_score['precision'] = np.mean(bert_score['precision'])
        bert_score['recall'] = np.mean(bert_score['recall'])
        bert_score['f1'] = np.mean(bert_score['f1'])
        torch.cuda.empty_cache()

        threshold = 300
        for i in range(len(prediction_list)):
            if len(prediction_list[i].split()) > threshold:
                pred = prediction_list[i].split()
                pred = pred[:threshold]
                prediction_list[i] = ' '.join(pred)
        
        for i in range(len(label_list)):
            if len(label_list[i].split()) > threshold:
                label = label_list[i].split()
                label = label[:threshold]
                label_list[i] = ' '.join(label)

        from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
        config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12')
        model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12').to(device)
        tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')

        model.eval()
        res = []

        with torch.no_grad():
            for i in range(len(label_list)):
                inputs = tokenizer(label_list[i], prediction_list[i], padding=True, return_tensors='pt', max_length=512, truncation=True).to(device)
                try:
                    tmp = model(**inputs).logits.flatten()
                except:
                    tmp = ''
                    num_skipped += 1
                    continue
                res.append(tmp.cpu())
        bleurt_score = np.mean(res)
        print('{"recall": %.3f, "precision": %.3f, "f1": %.3f, "bleurt": %.3f, "#invalid": %d}' % (bert_score['recall'], bert_score['precision'], bert_score['f1'], bleurt_score, num_skipped))

    #evaluation for ranking tasks
    elif (task == 'Query_Product_Ranking'):
        import numpy as np

        def DCG(score_list):

            dcg = 0
            for i in range(len(score_list)):
                dcg += (2 ** score_list[i] - 1) / (np.log2(i + 2))
            return dcg
        
        score_mapping = {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0}
        label2score = {}

        option_labels_list = json.load(open('ECInstruct/Query_Product_Ranking/IND_Diverse_Instruction/label.json', 'r'))
        counter = 0
        for option_labels in option_labels_list:
            label2score[counter] = {}
            option = 'A'
            for option_label in option_labels:
                label2score[counter][option] = option_label
                option = chr(ord(option) + 1)
            counter += 1

        total_ndcg = 0
        skipped = 0
        for i in range(len(prediction_list)):
            scores = []
            ranks = prediction_list[i].strip().split(',')
            for rank in ranks:
                try:
                    scores.append(score_mapping[label2score[i][rank[0]]])
                except:
                    skipped += 1
                    continue
            
            ideal_scores = sorted(scores, reverse=True)

            dcg  = DCG(scores)
            idcg = DCG(ideal_scores)
            total_ndcg += (dcg / (idcg + 1e-6))
        
        avg_ndcg = total_ndcg / len(label_list)
        print('{"NDCG": %.3f, "#invalid": %d}' % (avg_ndcg, skipped))

if __name__ == '__main__':
    fire.Fire(main)
