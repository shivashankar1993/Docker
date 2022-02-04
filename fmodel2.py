from flask import Flask, jsonify, request
from flask_restful import Resource, Api
# import pickle
import numpy as np
# import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import cv2
import time
# import glob
# import os.path

# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)



# # making a class for a particular resource
# # the get, post methods correspond to get and post requests
# # they are automatically mapped by flask_restful.


class Predict(Resource):
    def post(self):
        '''
        This function takes the following inputs:
        1) model_name - pickle file to be used as model input
        2) config_name - config csv file format for loading the configurations
        3) model inputs - variable name (parameter) along with parameter value
        
        Post reading the inputs - model file is utilized to forecast the output.
        '''

        #=========REQUIREMENT=========#
        input_format = request.form.get("input_format")


        if input_format == 'value' :

            print('\n\n\nRequired Variables : \n -> input_format \n -> model \n -> config \n -> model_variabes \n\n\n')
            
            #Loading Models
            #=========REQUIREMENT=========#
            try:
                model_name = request.form.get("model")
                model_filename = model_name + ".pkl"
                model = joblib.load('source_model/' + model_filename )
            except:
                print("="*20,"Model Not Supplied","="*20,"\n\n\n")


            #loading configrations
            #=========REQUIREMENT=========#
            try:
                config_name = request.form.get("config")       
                config_filename = config_name + ".csv"
                config = pd.read_csv('source_config/' + config_filename)
            except:
                print("="*20,"Config file Not Supplied","="*20,"\n\n\n")

            print(config)
            print(model)

            #Definition
            dict={} 
            try:   
                for i in range(len(config)):
                    dict["var{0}".format(i+1)] = float(request.form.get(config.iloc[i,0]))
            except:
                print("="*20,"Issue in Config parameters","="*20,"\n\n\n")
                
            values = dict.values()
            values_list = list(values)
            print(values_list)
            prediction = model.predict([values_list])
            output = jsonify(prediction.tolist())

        elif input_format == 'file' :

            print('\n\n\nRequired Variables : \n -> input_format \n -> model \n -> file \n\n\n')

            #Loading Models
            #=========REQUIREMENT=========#
            try:
                model_name = request.form.get("model")
                model_filename = model_name + ".pkl"
                model = joblib.load('source_model/' + model_filename )
            except:
                print("="*20,"Model Not Supplied","="*20,"\n\n\n")

            #=========REQUIREMENT=========#
            try:
                data = request.files["file"]
                data = pd.read_csv(data)
            except:
                print("="*20,"Input Data File Not Supplied","="*20,"\n\n\n")

            prediction = model.predict(data)
            output = jsonify(prediction.tolist())

        
        elif input_format == 'text' : 

            print('\n\n\nRequired Variables : \n -> input_format \n -> model \n -> text \n\n\n')       
            #Loading Models
            #=========REQUIREMENT=========#
            try:
                model_name = request.form.get("model")
                model_filename = model_name + ".pkl"
                model = joblib.load('source_model/' + model_filename )
            except:
                print("="*20,"Model Not Supplied","="*20,"\n\n\n")

            #Load count vector from disk
            cv = joblib.load(open('source_model/model_sentiment_transform.pkl','rb'))
            #Load the vocabulary
            words = joblib.load(open('source_model/model_sentiment_vocabulary.pkl','rb'))

            #loading text data from user
            #=========REQUIREMENT=========#
            try:
                text = request.form.get("text")
                data = [text]
            except:
                print("="*20,"Input Text Not Supplied","="*20,"\n\n\n")

            countVect = CountVectorizer(vocabulary=words)
            sentence = countVect.transform(data).toarray()

            # vectorize the user's query and make a prediction
            review_prediction = model.predict(sentence)[0]
            review_prediction_probablity = model.predict_proba(sentence)
                    
            # Output either 'Negative' or 'Positive' along with the score
            if review_prediction == 0:
                pred_text = 'Negative'
                confidence = review_prediction_probablity[0][0]
            else:
                pred_text = 'Positive'
                confidence = review_prediction_probablity[0][1]

            # create JSON object
            output = {'prediction': pred_text, 'confidence': confidence}   

                    
        elif input_format == 'image' : 
            print('\n\n\nRequired Variables : \n -> input_format \n -> model - prototxt, caffemodel, npy \n -> image \n\n\n')

            #=========REQUIREMENT=========# 
            try:
                prototxt = request.form.get("prototxt")
                #=========REQUIREMENT=========#
                caffemodel = request.form.get("caffemodel")
                #=========REQUIREMENT=========#
                npy = request.form.get("npy")       
                net = cv2.dnn.readNetFromCaffe('./source_model/' + prototxt,'./source_model/' + caffemodel)
                pts = np.load('./source_model/' + npy)
            except:
                print("="*20,"Models Not Supplied","="*20,"\n\n\n")
            class8 = net.getLayerId("class8_ab")
            conv8 = net.getLayerId("conv8_313_rh")
            pts = pts.transpose().reshape(2,313,1,1)
            net.getLayer(class8).blobs = [pts.astype("float32")]
            net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')]

            #=========REQUIREMENT=========#
            filestr = request.files['image'].read()
            #convert string data to numpy array
            npimg = np.fromstring(filestr, np.uint8)
            # convert numpy array to image
            try:
                image = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
            except:
                print("="*20,"Image Not Supplied","="*20,"\n\n\n")
            
            scaled = image.astype("float32")/255.0
            lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)


            resized = cv2.resize(lab,(224,224))
            L = cv2.split(resized)[0]
            L -= 50


            net.setInput(cv2.dnn.blobFromImage(L))
            ab = net.forward()[0, :, :, :].transpose((1,2,0))

            ab = cv2.resize(ab, (image.shape[1],image.shape[0]))

            L = cv2.split(lab)[0]
            colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)

            colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized,0,1)

            colorized = (255 * colorized).astype("uint8")

            # cv2.imshow("Original",image)
            # cv2.imshow("Colorized",colorized)
            #cv2.imwrite('Original.png',image)
            
            timestr = time.strftime("%Y%m%d-%H%M%S")
            #cv2.imwrite('Colorized1.png' + timestr, colorized)
            cv2.imwrite('./output_files/Colorized_' + timestr + '.png' , colorized)

            #cv2.waitKey(0)
            output = 0

        return output



api.add_resource(Predict, '/predict')

# driver function
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=False)
