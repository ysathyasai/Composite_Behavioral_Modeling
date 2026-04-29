import os
import re
import warnings
from django.conf import settings
from django.db.models import Count, Q
from django.shortcuts import render, redirect, get_object_or_404

import ipaddress
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# Create your views here.
from Remote_User.models import ClientRegister_Model,identity_theft_detection,detection_ratio,detection_accuracy


def parse_account_id(account_id):
    parts = str(account_id).split('-')
    if len(parts) >= 5:
        src_ip, dst_ip, src_port, dst_port, proto = parts[:5]
    elif len(parts) == 4:
        src_ip, dst_ip, src_port, dst_port = parts
        proto = '0'
    else:
        src_ip = dst_ip = ''
        src_port = dst_port = proto = '0'

    def is_private(ip):
        try:
            return ipaddress.ip_address(ip).is_private
        except Exception:
            return False

    return {
        'src_port': int(src_port) if str(src_port).isdigit() else 0,
        'dst_port': int(dst_port) if str(dst_port).isdigit() else 0,
        'protocol': int(proto) if str(proto).isdigit() else 0,
        'src_private': is_private(src_ip),
        'dst_private': is_private(dst_ip),
    }


def is_text_value(value):
    return bool(value and re.search(r'[A-Za-z]', str(value)))


def validate_prediction_input(data):
    errors = []
    if not data.get('Account_Id') or '-' not in data['Account_Id'] or len(data['Account_Id'].split('-')) < 4:
        errors.append('Account Reference must use the expected segmented format.')
    if not data.get('Trans_Id'):
        errors.append('Transaction ID is required.')

    try:
        age = int(data.get('Age', ''))
        if age < 10 or age > 120:
            errors.append('Behavioral Age must be between 10 and 120.')
    except Exception:
        errors.append('Behavioral Age must be a whole number.')

    try:
        followers = int(data.get('Followers', ''))
        if followers < 0 or followers > 10000000:
            errors.append('Social Connections must be a positive whole number under 10 million.')
    except Exception:
        errors.append('Social Connections must be a whole number.')

    for field, label in [
        ('NAME_CONTRACT_TYPE', 'Contract Geometry'),
        ('GENDER', 'Neural Gender'),
        ('NAME_INCOME_TYPE', 'Income Channel'),
        ('NAME_FAMILY_STATUS', 'Kinship Status'),
    ]:
        value = data.get(field, '')
        if not is_text_value(value):
            errors.append(f'{label} must contain alphabetic text.')

    for field, label in [
        ('AMT_INCOME_TOTAL', 'Net Liquid Wealth'),
        ('AMT_CREDIT', 'Active Credit'),
        ('AMT_ANNUITY', 'Annuity Volume'),
        ('AMT_GOODS_PRICE', 'Asset Valuation'),
    ]:
        try:
            amount = float(data.get(field, ''))
            if amount < 0 or amount > 100000000:
                errors.append(f'{label} must be a non-negative value under 100 million.')
        except Exception:
            errors.append(f'{label} must be a valid number.')

    return errors


def login(request):
    if request.method == "POST" and 'submit1' in request.POST:
        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username, password=password)
            request.session["userid"] = enter.id
            return redirect('ViewYourProfile')
        except ClientRegister_Model.DoesNotExist:
            error_message = "Invalid username or password. Please try again."

    return render(request, 'RUser/login.html', {'error_message': error_message if 'error_message' in locals() else None})

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    if 'userid' not in request.session:
        return redirect('login')

    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id=userid)
    return render(request, 'RUser/ViewYourProfile.html', {'object': obj})


def logout(request):
    request.session.pop('userid', None)
    return redirect('login')


def Predict_Theft_Status(request):
    if 'userid' not in request.session:
        return redirect('login')

    if request.method == "POST":
        Account_Id=request.POST.get('Account_Id')
        Trans_Id=request.POST.get('Trans_Id')
        Age=request.POST.get('Age')
        Followers=request.POST.get('Followers')
        NAME_CONTRACT_TYPE=request.POST.get('NAME_CONTRACT_TYPE')
        GENDER=request.POST.get('GENDER')
        AMT_INCOME_TOTAL=request.POST.get('AMT_INCOME_TOTAL')
        AMT_CREDIT=request.POST.get('AMT_CREDIT')
        AMT_ANNUITY=request.POST.get('AMT_ANNUITY')
        AMT_GOODS_PRICE=request.POST.get('AMT_GOODS_PRICE')
        NAME_INCOME_TYPE=request.POST.get('NAME_INCOME_TYPE')
        NAME_FAMILY_STATUS=request.POST.get('NAME_FAMILY_STATUS')

        validation_warnings = validate_prediction_input({
            'Account_Id': Account_Id,
            'Trans_Id': Trans_Id,
            'Age': Age,
            'Followers': Followers,
            'NAME_CONTRACT_TYPE': NAME_CONTRACT_TYPE,
            'GENDER': GENDER,
            'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
            'AMT_CREDIT': AMT_CREDIT,
            'AMT_ANNUITY': AMT_ANNUITY,
            'AMT_GOODS_PRICE': AMT_GOODS_PRICE,
            'NAME_INCOME_TYPE': NAME_INCOME_TYPE,
            'NAME_FAMILY_STATUS': NAME_FAMILY_STATUS,
        })

        data_path = os.path.join(settings.BASE_DIR, 'Datasets.csv')
        df = pd.read_csv(data_path)
        df['results'] = df['Label'].astype(int)

        parsed = df['Account_Id'].apply(parse_account_id).apply(pd.Series)
        df = pd.concat([df, parsed], axis=1)

        df['credit_income_ratio'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['annuity_income_ratio'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['goods_income_ratio'] = df['AMT_GOODS_PRICE'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['followers_log'] = np.log1p(np.maximum(df['Followers'], 0))
        df['followers_neg'] = (df['Followers'] < 0).astype(int)

        feature_cols = [
            'Age', 'Followers', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
            'AMT_GOODS_PRICE', 'credit_income_ratio', 'annuity_income_ratio',
            'goods_income_ratio', 'followers_log', 'followers_neg',
            'src_port', 'dst_port', 'protocol', 'src_private', 'dst_private',
            'NAME_CONTRACT_TYPE', 'GENDER', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS'
        ]
        categorical_cols = ['NAME_CONTRACT_TYPE', 'GENDER', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'src_private', 'dst_private']
        numeric_cols = ['Age', 'Followers', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                        'credit_income_ratio', 'annuity_income_ratio', 'goods_income_ratio', 'followers_log',
                        'followers_neg', 'src_port', 'dst_port', 'protocol']

        X = df[feature_cols]
        y = df['results']

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols),
            ],
            remainder='drop'
        )

        models = [
            ('random_forest', Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=300, max_depth=14, class_weight='balanced_subsample', random_state=42))
            ])),
            ('gradient_boosting', Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, random_state=42))
            ])),
            ('logistic', Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(max_iter=5000, class_weight='balanced', C=0.75, solver='liblinear', random_state=42))
            ])),
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )

        best_model = None
        best_score = 0.0
        for name, model in models:
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            if score > best_score:
                best_score = score
                best_model = model

        if best_model is None:
            best_model = models[0][1]

        model_name = [name for name, model in models if model is best_model][0]
        print(f"Best model: {model_name} accuracy={best_score*100:.2f}%")

        account_features = parse_account_id(Account_Id)
        input_data = {
            'Age': float(Age or 0),
            'Followers': float(Followers or 0),
            'AMT_INCOME_TOTAL': float(AMT_INCOME_TOTAL or 0),
            'AMT_CREDIT': float(AMT_CREDIT or 0),
            'AMT_ANNUITY': float(AMT_ANNUITY or 0),
            'AMT_GOODS_PRICE': float(AMT_GOODS_PRICE or 0),
            'credit_income_ratio': float(AMT_CREDIT or 0) / (float(AMT_INCOME_TOTAL or 0) + 1),
            'annuity_income_ratio': float(AMT_ANNUITY or 0) / (float(AMT_INCOME_TOTAL or 0) + 1),
            'goods_income_ratio': float(AMT_GOODS_PRICE or 0) / (float(AMT_INCOME_TOTAL or 0) + 1),
            'followers_log': np.log1p(max(float(Followers or 0), 0)),
            'followers_neg': 1 if float(Followers or 0) < 0 else 0,
            'src_port': account_features['src_port'],
            'dst_port': account_features['dst_port'],
            'protocol': account_features['protocol'],
            'src_private': account_features['src_private'],
            'dst_private': account_features['dst_private'],
            'NAME_CONTRACT_TYPE': NAME_CONTRACT_TYPE or '',
            'GENDER': GENDER or '',
            'NAME_INCOME_TYPE': NAME_INCOME_TYPE or '',
            'NAME_FAMILY_STATUS': NAME_FAMILY_STATUS or ''
        }
        input_df = pd.DataFrame([input_data])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Found unknown categories in columns')
            predict_text = best_model.predict(input_df)

        prediction = int(predict_text[0])
        if prediction == 0:
            val = 'No Theft or Fraud Found'
        else:
            val = 'Theft or Fraud Found'

        identity_theft_detection.objects.create(
            Account_Id=Account_Id,
            Trans_Id=Trans_Id,
            Age=int(Age) if Age else 0,
            Followers=int(Followers) if Followers else 0,
            NAME_CONTRACT_TYPE=NAME_CONTRACT_TYPE,
            GENDER=GENDER,
            AMT_INCOME_TOTAL=float(AMT_INCOME_TOTAL) if AMT_INCOME_TOTAL else 0.0,
            AMT_CREDIT=float(AMT_CREDIT) if AMT_CREDIT else 0.0,
            AMT_ANNUITY=float(AMT_ANNUITY) if AMT_ANNUITY else 0.0,
            AMT_GOODS_PRICE=float(AMT_GOODS_PRICE) if AMT_GOODS_PRICE else 0.0,
            NAME_INCOME_TYPE=NAME_INCOME_TYPE,
            NAME_FAMILY_STATUS=NAME_FAMILY_STATUS,
            Prediction=val
        )

        context = {'objs': val}
        if validation_warnings:
            context['validation_warnings'] = validation_warnings
        return render(request, 'RUser/Predict_Theft_Status.html', context)
    return render(request, 'RUser/Predict_Theft_Status.html')


def Predict_Test_Inputs(request):
    if 'userid' not in request.session:
        return redirect('login')

    data_path = os.path.join(settings.BASE_DIR, 'Predict_Test_Inputs.csv')
    try:
        df = pd.read_csv(data_path, dtype=str)
    except FileNotFoundError:
        df = pd.DataFrame()

    samples = df.to_dict(orient='records') if not df.empty else []
    return render(request, 'RUser/Predict_Test_Inputs.html', {'samples': samples})


