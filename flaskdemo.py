from flask import Flask,request,render_template
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
diabetes = pd.read_csv('diabetes.csv')
diabetes_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_mod[feature_names]
y = diabetes_mod.Outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Outcome, random_state=0)

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html',title='Home')


@app.route('/form')
def form():
     return render_template('form.html',title='Form')



@app.route('/result', methods=['POST'])
def result():
    age=gl=bp=st=insu=bmi=pf=pg=0
    name=request.form['Name']
    age=request.form['age']
    gl=request.form['gl']
    bp=request.form['bp']
    pg=request.form['pgs']
    st=request.form['st']
    insu=request.form['in']
    bmi=request.form['bmi']
    pf=request.form['pf']
    
    data=[[pg,gl,bp,st,insu,bmi,pf,age]]
    model = KNeighborsClassifier()
    model.fit(X_test,y_test)
    output=int(model.predict(data))
    return render_template('result.html',title='Result', data=output, nm=name)
   
 
if __name__=='__main__':
    app.run(debug=True)