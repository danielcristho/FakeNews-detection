import pickle

#function to run for prediction
def detectingFakeNews(var):
    #retrieving the best model for prediction call
    loadModel = pickle.load(open('model/final_model.sav', 'rb'))
    prediction = loadModel.predict([var])
    prob = loadModel.predict_proba([var])

    return prediction, prob