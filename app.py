# Importing essential libraries
from flask import Flask, render_template, request,session,jsonify
import requests
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import google.generativeai as genai

app = Flask(__name__)

genai.configure(api_key="AIzaSyBt4aQFYIRbnzpd2jZT1GUP_O0y8gkvKYY")


newsapi_key = "899c9aee01ce4e3f9a347f3a98e7cc23"
query = '"Disease " OR "ICMR "  OR "Medical health"   '  # Search for mentions of WHO in the articles
url = f"https://newsapi.org/v2/everything?q={query}&apiKey={newsapi_key}"

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/register',methods=['GET','POST'])

def registeration():
    city = ['Abohar', 'Agartala', 'Agra', 'Ahmadnagar', 'Ahmedabad', 'Ahmednagar', 'Aizawl', 'Ajmer', 'Akola', 'Aligarh', 'Allahabad', 'Allepy', 'Ambala Cantt', 'Amravati', 'Amritsar', 'Anand', 'Anantapur', 'Arrah', 'Asansol', 'Aurangabad', 'Azamgarh', 'Bahadurgarh', 'Baharampur', 'Balasore', 'Balurghat', 'Banda', 'Bangalore', 'Bangalore', 'Bangalore', 'Bankura', 'Baramati', 'Bareilly', 'Barnala', 'Barshi', 'Baruipur', 'Basirhat', 'Basti', 'Batala', 'Bathinda', 'Beawar', 'Begusarai', 'Belgaum', 'Bellary', 'Berhampur', 'Bhagalpur', 'Bharatpur', 'Bhaurach', 'Bhavnagar', 'Bhind', 'Bhiwani', 'Bhopal', 'Bhubaneswar', 'Bid', 'Bihar Shariff', 'Bijapur', 'Bijnor', 'Bikaner', 'Bilaspur', 'Bogra', 'Bokaro', 'Bulandshahar', 'Burdwan', 'Calicut', 'Chandigarh', 'Chandrapur', 'Chapra', 'Chennai', 'Chennai', 'Chhindwara', 'Chittagong', 'Cochin', 'Coimbatore', 'Cooch Bihar', 'Cuttack', 'Daltonganj', 'Davangere', 'Dehradun', 'Delhi', 'Delhi', 'Delhi', 'Delhi', 'Delhi', 'Dhaka', 'Dhaka', 'Dhaka', 'Dhanbad', 'Dharmapuri', 'Dibrugarh', 'Dimapur', 'Durg', 'Durgapur', 'Erode', 'Faizabad', 'Faridabad', 'Faridkot', 'Farrukhabad', 'Firozabad', 'Gandhinagar', 'Gaya', 'Ghaziabad', 'Godhra', 'Gondia', 'Gopalganj', 'Gulbarga', 'Guna', 'Guntur', 'Gurgaon', 'Guwahati', 'Gwalior', 'Habra', 'Haldia', 'Haldwani', 'Hanumangarh', 'Hardwar', 'Hazaribagh', 'Hissar', 'Hoshiarpur', 'Hospet', 'Howrah', 'Hubli', 'Hyderabad', 'Hyderabad', 'Ichalkaranji', 'Indore', 'Itanagar', 'Jabalpur', 'Jaipur', 'Jaisalmer', 'Jalandhar', 'Jalgaon', 'Jalpaiguri', 'Jammu', 'Jamshedpur', 'Jaunpur', 'Jehanabad', 'Jeypur', 'Jhansi', 'Jharsuguda', 'Jhunjhunu', 'Jind', 'Jodhpur', 'Jorhat', 'Kaithal', 'Kakinada', 'Kannur', 'Kanpur', 'Karimnagar', 'Karnal', 'Kashipur', 'Katihar', 'Khammam', 'Khandwa', 'Khanna', 'Khargone', 'Khulna', 'Kolhapur', 'Kolkata', 'Kolkata', 'Kollam', 'Korba', 'Kota', 'Kottayam', 'Kullu', 'Kurnool', 'Kurukshetra', 'Latur', 'Lucknow', 'Ludhiana', 'Madurai', 'Mahesana', 'Mandalay', 'Mandi', 'Mandsaur', 'Mangalore', 'Mapusa', 'Mathura', 'Meerut', 'Moga', 'Mohali', 'Moradabad', 'Motihari', 'Mughalsarai', 'Muktsar', 'Mumbai', 'Mumbai', 'Mumbai', 'Mumbai-Kalyan', 'Munger', 'Muvattupuzha', 'Muzaffarnagar', 'Muzaffarpur', 'Mysore', 'Nagaon', 'Nagercoil', 'Nagpur', 'Nalgonda', 'Namakkal', 'Nanded', 'Nandyal', 'Narnaul', 'Nashik', 'Nellore', 'Mumbai', 'Neyveli', 'Nizamabad', 'Noida', 'Ongole', 'Palakkad', 'Panchkula', 'Panipat', 'Pathanamthitta', 'Pathankot', 'Patiala', 'Patna', 'Pollachi', 'Pondicherry', 'Pudukkottai', 'Pune', 'Purnea', 'Purulia', 'Rae Bareli', 'Raichur', 'Raipur', 'Rajahmundry', 'Rajshahi', 'Rampur', 'Ranchi', 'Ratlam', 'Raxaul Bazar', 'Rewa', 'Rewari', 'Rohtak', 'Roorkee', 'Rourkela', 'Rudrapur', 'Rupnagar', 'Sagar', 'Saharanpur', 'Saharsa', 'Salem', 'Samastipur', 'Sambalpur', 'Sangli', 'Sasaram', 'Satara', 'Satna', 'Serampore', 'Shahjahanpur', 'Shillong', 'Shimla', 'Shimoga', 'Shivpuri', 'Sikar', 'Siliguri', 'Sirsa', 'Sitamarhi', 'Sitapur', 'Siwan', 'Solan', 'Solapur', 'Sonipat', 'Sri Ganganagar', 'Srikakulam', 'Sultanpur', 'Surat', 'Surendranagar', 'Suri', 'Tarn-Taran', 'Tezpur', 'Thanjavur', 'Thrissur', 'Tinsukia', 'Tirunelveli', 'Tirupati', 'Tiruppur', 'Tiruvannamalai', 'Trichy', 'Tumkur', 'Tuticorin', 'Udaipur', 'Udaipur', 'Udupi', 'Ujjain', 'Vadodara', 'Valsad', 'Vapi', 'Varanasi', 'Vidisha', 'Vijayawada', 'Visakhapatnam', 'Warangal', 'Yamunanagar', 'Yavatmal', ]
    city=sorted(city)
    return render_template('Register.html',city=city)



@app.route('/login',methods=['GET','POST'])
def log():
     return render_template('Register.html')

# Define the route for prediction
conn= sqlite3.connect('database.db')
conn.execute('''CREATE TABLE IF NOT EXISTS users 
                 (name text, sex text, phone text, city text, email text, password text)''')
#result for registration 

myreg=set()
@app.route('/HeathHome',methods=['GET','POST'])
def Register():
    render_template("Register.html")
    if request.method == 'POST':
         name = request.form.get('name')
         sex = request.form.get('sex')
         city = request.form.get('city')
         phone = request.form.get('phone')

         email = request.form.get('email')
         password = request.form.get('password')
    myreg.add(name)
    myreg.add(sex)
    myreg.add(city)
    myreg.add(phone)
    myreg.add(email)
    print(myreg)
    conn= sqlite3.connect('database.db')
    conn.cursor()
    conn.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)", (name, sex, city,phone, email, password))
    conn.commit()
    conn.close()
    return render_template('Register.html')
myset=set()
@app.route('/main', methods=['GET', 'POST'])
def login():
    render_template("Register.html")
    error = None
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email=? and password=?", (email, password))
    user = cursor.fetchone()
    conn.close()
    if user:
        myset.add(email)
        myset.add(password)
        return render_template('HeathHome.html',user=myset)
    else:
         error = 'Invalid Credentials. Please try again.'
         return render_template('Register.html',error=error)


@app.route('/logout',methods=['GET','POST'])
def logout():

    myset.clear()
    myreg.clear()
    predictresult.clear()
    print(myreg)
    return render_template('HeathHome.html')


@app.route('/')
def home():
	return render_template('HeathHome.html',user=myset)
@app.route('/disease',methods=['GET','POST'])

def index():
    print(myset)
    if(len(myset)==0):
        return render_template('aakash.html')
    else:
        l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
	 'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
	 'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
	 'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
	 'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
	 'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
	 'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
	 'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
	 'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
	 'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
	 'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
	 'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
	 'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
	 'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
	 'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
	 'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
	 'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
	 'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
	 'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
	 'yellow_crust_ooze']
        l1=sorted(l1)
        return render_template('disease.html', l1=l1)

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
	 'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
	 'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
	 'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
	 'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
	 'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
	 'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
	 'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
	 'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
	 'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
	 'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
	 'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
	 'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
	 'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
	 'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
	 'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
	 'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
	 'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
	 'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
	 'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("templates/testing.csv")
df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("training.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------

gnb = MultinomialNB()
gnb = gnb.fit(X, y)
# Define the route for prediction
predictresult=set()
@app.route('/result',methods=['GET','POST'])

def predict():
    try:
        render_template('disease.html')
        disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction','Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',' Migraine','Cervical spondylosis',
             'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']
        clf4 = RandomForestClassifier()
        clf4 = clf4.fit(X,np.ravel(y))
        # calculating accuracy-------------------------------------------------------------------
        from sklearn.metrics import accuracy_score
        y_pred=clf4.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------
    # 
        if request.method == 'POST':
            sy1 = request.form.get('sy1')
            sy2 = request.form.get('sy2')
            sy3 = request.form.get('sy3')
            sy4 = request.form.get('sy4')
            sy5 = request.form.get('sy5')
        psymptoms = [sy1,sy2,sy3,sy4,sy5]
        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = clf4.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
        
        if (h=='yes'):
              # pred=disease[a]
            predictresult.add(disease[a])
            print(predictresult)
            return render_template('result.html', result=list(predictresult)[0],user=myset)
    except:
        return render_template('result.html', result=predictresult,user=myset)

@app.route('/about')
def about():
    city = ['Abohar', 'Agartala', 'Agra', 'Ahmadnagar', 'Ahmedabad', 'Ahmednagar', 'Aizawl', 'Ajmer', 'Akola', 'Aligarh', 'Allahabad', 'Allepy', 'Ambala Cantt', 'Amravati', 'Amritsar', 'Anand', 'Anantapur', 'Arrah', 'Asansol', 'Aurangabad', 'Azamgarh', 'Bahadurgarh', 'Baharampur', 'Balasore', 'Balurghat', 'Banda', 'Bangalore', 'Bangalore', 'Bangalore', 'Bankura', 'Baramati', 'Bareilly', 'Barnala', 'Barshi', 'Baruipur', 'Basirhat', 'Basti', 'Batala', 'Bathinda', 'Beawar', 'Begusarai', 'Belgaum', 'Bellary', 'Berhampur', 'Bhagalpur', 'Bharatpur', 'Bhaurach', 'Bhavnagar', 'Bhind', 'Bhiwani', 'Bhopal', 'Bhubaneswar', 'Bid', 'Bihar Shariff', 'Bijapur', 'Bijnor', 'Bikaner', 'Bilaspur', 'Bogra', 'Bokaro', 'Bulandshahar', 'Burdwan', 'Calicut', 'Chandigarh', 'Chandrapur', 'Chapra', 'Chennai', 'Chennai', 'Chhindwara', 'Chittagong', 'Cochin', 'Coimbatore', 'Cooch Bihar', 'Cuttack', 'Daltonganj', 'Davangere', 'Dehradun', 'Delhi', 'Delhi', 'Delhi', 'Delhi', 'Delhi', 'Dhaka', 'Dhaka', 'Dhaka', 'Dhanbad', 'Dharmapuri', 'Dibrugarh', 'Dimapur', 'Durg', 'Durgapur', 'Erode', 'Faizabad', 'Faridabad', 'Faridkot', 'Farrukhabad', 'Firozabad', 'Gandhinagar', 'Gaya', 'Ghaziabad', 'Godhra', 'Gondia', 'Gopalganj', 'Gulbarga', 'Guna', 'Guntur', 'Gurgaon', 'Guwahati', 'Gwalior', 'Habra', 'Haldia', 'Haldwani', 'Hanumangarh', 'Hardwar', 'Hazaribagh', 'Hissar', 'Hoshiarpur', 'Hospet', 'Howrah', 'Hubli', 'Hyderabad', 'Hyderabad', 'Ichalkaranji', 'Indore', 'Itanagar', 'Jabalpur', 'Jaipur', 'Jaisalmer', 'Jalandhar', 'Jalgaon', 'Jalpaiguri', 'Jammu', 'Jamshedpur', 'Jaunpur', 'Jehanabad', 'Jeypur', 'Jhansi', 'Jharsuguda', 'Jhunjhunu', 'Jind', 'Jodhpur', 'Jorhat', 'Kaithal', 'Kakinada', 'Kannur', 'Kanpur', 'Karimnagar', 'Karnal', 'Kashipur', 'Katihar', 'Khammam', 'Khandwa', 'Khanna', 'Khargone', 'Khulna', 'Kolhapur', 'Kolkata', 'Kolkata', 'Kollam', 'Korba', 'Kota', 'Kottayam', 'Kullu', 'Kurnool', 'Kurukshetra', 'Latur', 'Lucknow', 'Ludhiana', 'Madurai', 'Mahesana', 'Mandalay', 'Mandi', 'Mandsaur', 'Mangalore', 'Mapusa', 'Mathura', 'Meerut', 'Moga', 'Mohali', 'Moradabad', 'Motihari', 'Mughalsarai', 'Muktsar', 'Mumbai', 'Mumbai', 'Mumbai', 'Mumbai-Kalyan', 'Munger', 'Muvattupuzha', 'Muzaffarnagar', 'Muzaffarpur', 'Mysore', 'Nagaon', 'Nagercoil', 'Nagpur', 'Nalgonda', 'Namakkal', 'Nanded', 'Nandyal', 'Narnaul', 'Nashik', 'Nellore', 'Mumbai', 'Neyveli', 'Nizamabad', 'Noida', 'Ongole', 'Palakkad', 'Panchkula', 'Panipat', 'Pathanamthitta', 'Pathankot', 'Patiala', 'Patna', 'Pollachi', 'Pondicherry', 'Pudukkottai', 'Pune', 'Purnea', 'Purulia', 'Rae Bareli', 'Raichur', 'Raipur', 'Rajahmundry', 'Rajshahi', 'Rampur', 'Ranchi', 'Ratlam', 'Raxaul Bazar', 'Rewa', 'Rewari', 'Rohtak', 'Roorkee', 'Rourkela', 'Rudrapur', 'Rupnagar', 'Sagar', 'Saharanpur', 'Saharsa', 'Salem', 'Samastipur', 'Sambalpur', 'Sangli', 'Sasaram', 'Satara', 'Satna', 'Serampore', 'Shahjahanpur', 'Shillong', 'Shimla', 'Shimoga', 'Shivpuri', 'Sikar', 'Siliguri', 'Sirsa', 'Sitamarhi', 'Sitapur', 'Siwan', 'Solan', 'Solapur', 'Sonipat', 'Sri Ganganagar', 'Srikakulam', 'Sultanpur', 'Surat', 'Surendranagar', 'Suri', 'Tarn-Taran', 'Tezpur', 'Thanjavur', 'Thrissur', 'Tinsukia', 'Tirunelveli', 'Tirupati', 'Tiruppur', 'Tiruvannamalai', 'Trichy', 'Tumkur', 'Tuticorin', 'Udaipur', 'Udaipur', 'Udupi', 'Ujjain', 'Vadodara', 'Valsad', 'Vapi', 'Varanasi', 'Vidisha', 'Vijayawada', 'Visakhapatnam', 'Warangal', 'Yamunanagar', 'Yavatmal', ]
    city=sorted(city)
    return render_template('Predicare.html',user=myset,city=city)

def get_user_details(email, password):
    conn = sqlite3.connect('database.db')  # Replace 'database.db' with your database name
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email=? AND password=?', (email, password))
    user = cursor.fetchone()
    conn.close()
    return user
@app.route('/aakash')
def aaka():
    email = list(myset)[0]
    password =list(myset)[1]
    print(email,password)
    user = get_user_details(email, password)
    name = user  # Extract the name from the fetched user details
    print(name)
    return render_template('Myaccount.html',user=user)

conn.execute('''CREATE TABLE IF NOT EXISTS feedback 
                 (name text, email text, state text, city text,suggestion text)''')

@app.route('/feed',methods=['GET','POST'])
def feedback():
    render_template("Predicare.html")
    citydata = ['Abohar', 'Agartala', 'Agra', 'Ahmadnagar', 'Ahmedabad', 'Ahmednagar', 'Aizawl', 'Ajmer', 'Akola', 'Aligarh', 'Allahabad', 'Allepy', 'Ambala Cantt', 'Amravati', 'Amritsar', 'Anand', 'Anantapur', 'Arrah', 'Asansol', 'Aurangabad', 'Azamgarh', 'Bahadurgarh', 'Baharampur', 'Balasore', 'Balurghat', 'Banda', 'Bangalore', 'Bangalore', 'Bangalore', 'Bankura', 'Baramati', 'Bareilly', 'Barnala', 'Barshi', 'Baruipur', 'Basirhat', 'Basti', 'Batala', 'Bathinda', 'Beawar', 'Begusarai', 'Belgaum', 'Bellary', 'Berhampur', 'Bhagalpur', 'Bharatpur', 'Bhaurach', 'Bhavnagar', 'Bhind', 'Bhiwani', 'Bhopal', 'Bhubaneswar', 'Bid', 'Bihar Shariff', 'Bijapur', 'Bijnor', 'Bikaner', 'Bilaspur', 'Bogra', 'Bokaro', 'Bulandshahar', 'Burdwan', 'Calicut', 'Chandigarh', 'Chandrapur', 'Chapra', 'Chennai', 'Chennai', 'Chhindwara', 'Chittagong', 'Cochin', 'Coimbatore', 'Cooch Bihar', 'Cuttack', 'Daltonganj', 'Davangere', 'Dehradun', 'Delhi', 'Delhi', 'Delhi', 'Delhi', 'Delhi', 'Dhaka', 'Dhaka', 'Dhaka', 'Dhanbad', 'Dharmapuri', 'Dibrugarh', 'Dimapur', 'Durg', 'Durgapur', 'Erode', 'Faizabad', 'Faridabad', 'Faridkot', 'Farrukhabad', 'Firozabad', 'Gandhinagar', 'Gaya', 'Ghaziabad', 'Godhra', 'Gondia', 'Gopalganj', 'Gulbarga', 'Guna', 'Guntur', 'Gurgaon', 'Guwahati', 'Gwalior', 'Habra', 'Haldia', 'Haldwani', 'Hanumangarh', 'Hardwar', 'Hazaribagh', 'Hissar', 'Hoshiarpur', 'Hospet', 'Howrah', 'Hubli', 'Hyderabad', 'Hyderabad', 'Ichalkaranji', 'Indore', 'Itanagar', 'Jabalpur', 'Jaipur', 'Jaisalmer', 'Jalandhar', 'Jalgaon', 'Jalpaiguri', 'Jammu', 'Jamshedpur', 'Jaunpur', 'Jehanabad', 'Jeypur', 'Jhansi', 'Jharsuguda', 'Jhunjhunu', 'Jind', 'Jodhpur', 'Jorhat', 'Kaithal', 'Kakinada', 'Kannur', 'Kanpur', 'Karimnagar', 'Karnal', 'Kashipur', 'Katihar', 'Khammam', 'Khandwa', 'Khanna', 'Khargone', 'Khulna', 'Kolhapur', 'Kolkata', 'Kolkata', 'Kollam', 'Korba', 'Kota', 'Kottayam', 'Kullu', 'Kurnool', 'Kurukshetra', 'Latur', 'Lucknow', 'Ludhiana', 'Madurai', 'Mahesana', 'Mandalay', 'Mandi', 'Mandsaur', 'Mangalore', 'Mapusa', 'Mathura', 'Meerut', 'Moga', 'Mohali', 'Moradabad', 'Motihari', 'Mughalsarai', 'Muktsar', 'Mumbai', 'Mumbai', 'Mumbai', 'Mumbai-Kalyan', 'Munger', 'Muvattupuzha', 'Muzaffarnagar', 'Muzaffarpur', 'Mysore', 'Nagaon', 'Nagercoil', 'Nagpur', 'Nalgonda', 'Namakkal', 'Nanded', 'Nandyal', 'Narnaul', 'Nashik', 'Nellore', 'Mumbai', 'Neyveli', 'Nizamabad', 'Noida', 'Ongole', 'Palakkad', 'Panchkula', 'Panipat', 'Pathanamthitta', 'Pathankot', 'Patiala', 'Patna', 'Pollachi', 'Pondicherry', 'Pudukkottai', 'Pune', 'Purnea', 'Purulia', 'Rae Bareli', 'Raichur', 'Raipur', 'Rajahmundry', 'Rajshahi', 'Rampur', 'Ranchi', 'Ratlam', 'Raxaul Bazar', 'Rewa', 'Rewari', 'Rohtak', 'Roorkee', 'Rourkela', 'Rudrapur', 'Rupnagar', 'Sagar', 'Saharanpur', 'Saharsa', 'Salem', 'Samastipur', 'Sambalpur', 'Sangli', 'Sasaram', 'Satara', 'Satna', 'Serampore', 'Shahjahanpur', 'Shillong', 'Shimla', 'Shimoga', 'Shivpuri', 'Sikar', 'Siliguri', 'Sirsa', 'Sitamarhi', 'Sitapur', 'Siwan', 'Solan', 'Solapur', 'Sonipat', 'Sri Ganganagar', 'Srikakulam', 'Sultanpur', 'Surat', 'Surendranagar', 'Suri', 'Tarn-Taran', 'Tezpur', 'Thanjavur', 'Thrissur', 'Tinsukia', 'Tirunelveli', 'Tirupati', 'Tiruppur', 'Tiruvannamalai', 'Trichy', 'Tumkur', 'Tuticorin', 'Udaipur', 'Udaipur', 'Udupi', 'Ujjain', 'Vadodara', 'Valsad', 'Vapi', 'Varanasi', 'Vidisha', 'Vijayawada', 'Visakhapatnam', 'Warangal', 'Yamunanagar', 'Yavatmal', ]
    citydata=sorted(citydata)
   
    if request.method == 'POST':
         name = request.form.get('name')
         email = request.form.get('email')
         state = request.form.get('state')
         city = request.form.get('city')
         suggestion = request.form.get('suggestion')
    print(name,email,state,city,suggestion)
    conn= sqlite3.connect('database.db')
    conn.cursor()
    conn.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?)", (name, email, state, city, suggestion))
    conn.commit()
    conn.close()
    return render_template('Predicare.html',user=myset,city=citydata)

def get_ai_response_google(conversation):
    try:
        text = conversation
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(text)
        response_text = response.text
        return response_text
    except Exception as e:
        return {'error': f'AI API error: {str(e)}'}

# Route to render the HTML page

def chat():
    return render_template('ak.html')

# Route to handle user input and return AI response
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    ai_response = get_ai_response_google(user_input)
    return jsonify({'ai_response': ai_response})

print(get_ai_response_google("hi"))


@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/get_news_headlines')
def get_news_headlines():
    try:
        main_page = requests.get(url).json()
        articles = main_page.get("articles", [])

        headlines = []
        for article in articles:
            if "title" in article:
                headlines.append(article["title"])

        return jsonify(headlines[:20])  # Return the first 6 headlines
    except Exception as e:
        return jsonify([])

    
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)


