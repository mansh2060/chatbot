from nlp import tokenize,bagofwords
import numpy as np
import tensorflow as tf
import json
import pickle
import random


with open('inp_resp.json','r') as file:
    data=json.load(file)


with open("information.pkl","rb") as file:
    infor=pickle.load(file)

all_tokens_list=infor['all_tokens_list']
category_list=infor['category_list']

model=tf.keras.models.load_model('model.h5')

bot_name='Jarvis'
def get_message():
    print("Let's chat(Type quit to exit)       :")
    while True:
        message=input("You):        ")
        if message=='quit':
            break
        tokens=tokenize(message)
        vectors=bagofwords(tokens,all_tokens_list)
        vectors=np.array(vectors,dtype=np.float32)
        vectors=vectors.reshape(1,vectors.shape[0])

        logits=model(vectors)
        predictions=tf.nn.softmax(logits)
        predicted_array=np.argmax(predictions,axis=1)
        predicted_category=category_list[predicted_array[0]]
        count=0

        for category in data['chatbot_responses']:
            if category['category']==predicted_category:
                print(f"{bot_name}:{(random.choice(category['response']))}")
                count=count+1
                break
        
        if count==0:
            print(f'{bot_name}:{"I do not understand"}')


get_message()