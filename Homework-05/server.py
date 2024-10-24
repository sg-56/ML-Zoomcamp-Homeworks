from flask import Flask,request
import pickle,jsonify

app = Flask('churn')

with open('dv.bin','rb') as file:
    dv = pickle.load(file)



with open('model1.bin','rb') as file:
    model = pickle.load(file)



@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == "GET":
        return 'The server is up....'
    customer = request.get_json()
    print(customer)
    data = dv.transform(customer)
    score = model.predict_proba(data)[0][1]
    churn = score>0.5
    result = {
        'churn_probability':str(score),
        'churn':str(churn),
    }
    print(result)
    return result



if __name__=="__main__":
    app.run(debug=True,port=9696,host='0.0.0.0')
