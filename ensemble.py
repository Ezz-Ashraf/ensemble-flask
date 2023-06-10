import warnings

# Disable warnings
warnings.filterwarnings('ignore')


import os

import numpy as np
from numpy import array

from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from keras_preprocessing.sequence import pad_sequences

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel , XLNetTokenizer, XLNetForSequenceClassification

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import tensorflow as tf

import tensorflow_hub  as hub
import bert
from bert import tokenization
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
sys.path.append('models')

from models.official.nlp.data import classifier_data_lib
import os

# Set the environment variable to disable tokenization parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tf.gfile = tf.io.gfile

# Set the random seed for PyTorch and NumPy
torch.manual_seed(0)
np.random.seed(0)

class Classifier(nn.Module):
    def __init__(self, base_model, num_labels=6, dropout_prob=0.3):
        super(Classifier, self).__init__()
        
        self.base_model = base_model
        self.num_labels = num_labels
        
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(self.base_model.config.hidden_size, self.num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # take the pooled output (CLS token) of the last layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
        
Roberta_MODEL = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(Roberta_MODEL)
base_model = AutoModel.from_pretrained(Roberta_MODEL)
# Load the robeta model & the tokenizer
roberta_model = Classifier(base_model)
roberta_model.load_state_dict(torch.load('roberta_korean.pth',map_location=torch.device('cpu')))
label_list=[0,1,2,3,4,5] 
max_seq_len=64
batch_size=32
bert_layer=hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2',trainable=True)
vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()
BertTokenizer=tokenization.FullTokenizer(vocab_file,do_lower_case)
# Load pre-trained XLNet model for sequence classification
XLmodel = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=6)
XLmodel.load_state_dict(torch.load('xlnet_korean.pth',map_location=torch.device('cpu')))


def create_model():

    input_word_ids=tf.keras.layers.Input(shape=(max_seq_len,),dtype=tf.int32,name='input_word_ids')
    input_mask=tf.keras.layers.Input(shape=(max_seq_len,),dtype=tf.int32,name='input_mask')
    input_type_ids=tf.keras.layers.Input(shape=(max_seq_len,),dtype=tf.int32,name='input_type_ids')
    pooled_output,sequence_output=bert_layer([input_word_ids,input_mask,input_type_ids])
    drop=tf.keras.layers.Dropout(0.2)(pooled_output)
    output=tf.keras.layers.Dense(6,activation='softmax',name='output')(drop)
    model=tf.keras.Model(
        inputs={
        'input_word_ids':input_word_ids,
        'input_mask':input_mask,
        'input_type_ids':input_type_ids
        },
        outputs=output
    )
    return model

BertModel=create_model()
BertModel.load_weights("my_weights/my_weights") 

def predict(sample):
   
        

    labels=['Anxiety','BPD','autism','bipolar','depression','schizophrenia']

    def predict_sentence_Roberta(model, tokenizer, sentence):
        # Convert the sentence to input features
        inputs = tokenizer(sentence, return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Run the model and get the predicted probabilities
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            Roberta_predictions=output.detach().numpy().ravel()
        return Roberta_predictions







    def tokenize_inputs(text_list, tokenizer, num_embeddings=120):
        tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings-2], text_list))
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
        input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
        return input_ids

    def create_attn_masks(input_ids):
        attention_masks = []
        for seq in input_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
        return attention_masks



    def to_feature(text, label, label_list=label_list, max_seq_length=max_seq_len, tokenizer=BertTokenizer):
    
    
        example = classifier_data_lib.InputExample(guid = None,
                                                    text_a = text.numpy(), 
                                                    text_b = None, 
                                                    label = label.numpy())
        feature = classifier_data_lib.convert_single_example(0, example, label_list,
                                            max_seq_length, tokenizer)
        
        return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)


    def to_feature_map(text, label):
        input_ids, input_mask, segment_ids, label_id = tf.py_function(to_feature, inp=[text, label], 
                                        Tout=[tf.int32, tf.int32, tf.int32, tf.int32])

        # py_func doesn't set the shape of the returned tensors.
        input_ids.set_shape([max_seq_len])
        input_mask.set_shape([max_seq_len])
        segment_ids.set_shape([max_seq_len])
        label_id.set_shape([])

        x = {
                'input_word_ids': input_ids,
                'input_mask': input_mask,
                'input_type_ids': segment_ids
            }
        return (x, label_id)




    def bertPredict(sample_example):
 
        predicted_data = tf.data.Dataset.from_tensor_slices((sample_example, [0]*len(sample_example)))
        predicted_data = (predicted_data.map(to_feature_map).batch(1))
        prediction=BertModel.predict(predicted_data)
        return prediction

    XLtokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    input_id = tokenize_inputs(sample, XLtokenizer, num_embeddings=128)
    input_id = torch.tensor(input_id)
    attention_masks = create_attn_masks(input_id)
    attention_masks = torch.tensor(attention_masks)
    bertPredictions =bertPredict(sample)
    bertPredictions=bertPredictions[0]
    Roberta_predictions= predict_sentence_Roberta(roberta_model, tokenizer, sample[0])
    outputs = XLmodel(input_id, token_type_ids=None, attention_mask=attention_masks)

    XLNet_predictions=outputs[0].detach().numpy().ravel()

    result = [sum(prediction) for prediction in zip(XLNet_predictions, Roberta_predictions,bertPredictions)]

    label_index = result.index(max(result))
    return labels[label_index]