import numpy as np

class WrapperModel():
    def __init__(self,dataset,model,preprocessing):
        self.dataset = dataset
        self.model = model
        self.preprocessing = preprocessing
    def predict(self,input_embbeded):
        if len(input_embbeded.shape)==1:
            x_len = np.sum(np.sign(input_embbeded))
            text = self.dataset.build_text(input_embbeded[:x_len])
            return self.model.predict_proba(self.preprocessing.transform([text]))
        else:
            x_len = np.sum(np.sign(input_embbeded),axis=-1)
            text = []
            for i in range(0,len(x_len)):
                text.append(self.dataset.build_text(input_embbeded[i][:x_len[i]]))
            answers = self.model.predict_proba(self.preprocessing.transform(text))            
            return np.squeeze(answers)