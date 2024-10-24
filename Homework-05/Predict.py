import pickle

with open('dv.bin','rb') as file:
     vectorizer = pickle.load(file)
     
with open('model1.bin','rb') as file:
     model = pickle.load(file)
     
     
customer = {"job": "management", "duration": 400, "poutcome": "success"}

data = vectorizer.transform(customer)

probs = model.predict_proba(data)
score = probs[0][1]

print(f"Probability Of Getting A subscription : {score}")
