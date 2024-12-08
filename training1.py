import json
from nlp import tokenize,stem,bagofwords
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pickle

with open('inp_resp.json','r') as file:
    data=json.load(file)


category_list=[]
all_tokens_list=[]
cat_tok_list=[]
X=[]
y=[]

for element in data['chatbot_responses']:
    category=element['category']
    category_list.append(category)
    for user_input in element['user_input']:
        tokens=(tokenize(user_input))
        all_tokens_list.extend(tokens)
        cat_tok_list.append((category,tokens))
       

all_tokens_list=sorted(set(all_tokens_list))
category_list=sorted(set(category_list))
#print("Category_list        :",len(category_list))

#print(all_tokens_list)
#print(len(all_tokens_list))
stop_words=set(stopwords.words('english'))
ignored_list=['?','.',',','-']
all_tokens_list=[stem(word) for word in all_tokens_list if word not in ignored_list and  word not in stop_words]
#print(len(all_tokens_list))

for (category,tokens) in cat_tok_list:
    vectors=bagofwords(tokens,all_tokens_list)
    X.append(vectors)
    labels=category_list.index(category)
    y.append(labels)

X=np.array(X)
y=np.array(y)

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

input_size=len(X_train[0])
hidden_size=8
output_size=len(category_list)
batch_size=8
epochs=500

#print(X_train.shape) #(30,77)
#print(x_test.shape) # (8, 77)
#print(type(x_test))

model=keras.models.Sequential(
    [
    keras.layers.Dense(units=input_size,activation='relu',input_shape=(input_size, )),
    keras.layers.Dense(units=hidden_size,activation='relu'),
    keras.layers.Dense(units=output_size),
    ]
)

loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim=keras.optimizers.Adam(learning_rate=0.001)
metrics=['accuracy']

model.compile(loss=loss,optimizer=optim,metrics=metrics)
model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True)


probability_model=keras.models.Sequential
(
    [
    model,
    keras.layers.Softmax() 
    ]
)

x_test=np.array(x_test,dtype=np.float32)
logits=model(x_test)
predictions=tf.nn.softmax(logits)
#print(predictions)
pred0=predictions[0]
predicted_value=np.argmax(pred0)
predicted_class=category_list[predicted_value]
#print(predicted_class)

model=model.save('model.h5')

information={
    "all_tokens_list":all_tokens_list,
    "category_list":category_list,
}

with open("information.pkl","wb") as file:
    pickle.dump(information,file)
