
import os
from functools import wraps
from django.db.models import Count, Avg, Q
from django.shortcuts import render, redirect
import datetime
import ipaddress
import xlwt
from django.conf import settings
from django.http import HttpResponse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
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
        'src_port': int(src_port) if src_port.isdigit() else 0,
        'dst_port': int(dst_port) if dst_port.isdigit() else 0,
        'protocol': int(proto) if str(proto).isdigit() else 0,
        'src_private': is_private(src_ip),
        'dst_private': is_private(dst_ip),
    }


def serviceproviderlogin(request):
    if request.method == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password == "Admin":
            request.session['admin_authenticated'] = True
            detection_accuracy.objects.all().delete()
            return redirect('View_Remote_Users')
        else:
            error_message = "Invalid admin credentials. Access denied."

    return render(request, 'SProvider/serviceproviderlogin.html', {'error_message': error_message if 'error_message' in locals() else None})


def admin_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.session.get('admin_authenticated'):
            return redirect('serviceproviderlogin')
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def admin_logout(request):
    request.session.pop('admin_authenticated', None)
    return redirect('serviceproviderlogin')

@admin_required
def View_Theft_Status_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'No Theft or Fraud Found'
    print(kword)
    obj = identity_theft_detection.objects.all().filter(Q(Prediction=kword))
    obj1 = identity_theft_detection.objects.all()
    count = obj.count()
    count1 = obj1.count()
    if count1 > 0:
        ratio = (count / count1) * 100
        if ratio != 0:
            detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio12 = ""
    kword12 = 'Theft or Fraud Found'
    print(kword12)
    obj12 = identity_theft_detection.objects.all().filter(Q(Prediction=kword12))
    obj112 = identity_theft_detection.objects.all()
    count12 = obj12.count()
    count112 = obj112.count()
    if count112 > 0:
        ratio12 = (count12 / count112) * 100
        if ratio12 != 0:
            detection_ratio.objects.create(names=kword12, ratio=ratio12)


    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Theft_Status_Ratio.html', {'objs': obj})

@admin_required
def View_Remote_Users(request):
    obj = ClientRegister_Model.objects.all()
    return render(request, 'SProvider/View_Remote_Users.html', {'objects': obj})

@admin_required
def charts(request, chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/charts.html", {'form': chart1, 'chart_type': chart_type})

@admin_required
def charts1(request, chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/charts1.html", {'form': chart1, 'chart_type': chart_type})

@admin_required
def View_Prediction_Of_Theft_Status(request):
    obj = identity_theft_detection.objects.all()
    return render(request, 'SProvider/View_Prediction_Of_Theft_Status.html', {'list_objects': obj})

@admin_required
def likeschart(request, like_chart):
    charts = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/likeschart.html", {'form': charts, 'like_chart': like_chart})


@admin_required
def Statistical(request):
    accuracy_stats = detection_accuracy.objects.all().order_by('-ratio')
    avg_accuracy = accuracy_stats.aggregate(avg=Avg('ratio'))['avg'] if accuracy_stats.exists() else None

    total_predictions = identity_theft_detection.objects.count()
    fraud_ratios = []
    if total_predictions > 0:
        no_fraud = identity_theft_detection.objects.filter(Prediction='No Theft or Fraud Found').count()
        theft = identity_theft_detection.objects.filter(Prediction='Theft or Fraud Found').count()
        fraud_ratios = [
            {'names': 'No Theft or Fraud Found', 'ratio': (no_fraud / total_predictions) * 100, 'count': no_fraud},
            {'names': 'Theft or Fraud Found', 'ratio': (theft / total_predictions) * 100, 'count': theft},
        ]

    return render(request, 'SProvider/Statistical.html', {
        'accuracy_stats': accuracy_stats,
        'avg_accuracy': avg_accuracy,
        'fraud_ratios': fraud_ratios,
        'total_predictions': total_predictions,
    })


@admin_required
def Download_Predicted_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Predicted_Datasets.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = identity_theft_detection.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:

        row_num = row_num + 1

        ws.write(row_num, 0, my_row.Account_Id, font_style)
        ws.write(row_num, 1, my_row.Trans_Id, font_style)
        ws.write(row_num, 2, my_row.Age, font_style)
        ws.write(row_num, 3, my_row.Followers, font_style)
        ws.write(row_num, 4, my_row.NAME_CONTRACT_TYPE, font_style)
        ws.write(row_num, 5, my_row.GENDER, font_style)
        ws.write(row_num, 6, my_row.AMT_INCOME_TOTAL, font_style)
        ws.write(row_num, 7, my_row.AMT_CREDIT, font_style)
        ws.write(row_num, 8, my_row.AMT_ANNUITY, font_style)
        ws.write(row_num, 9, my_row.AMT_GOODS_PRICE, font_style)
        ws.write(row_num, 10, my_row.NAME_INCOME_TYPE, font_style)
        ws.write(row_num, 11, my_row.NAME_FAMILY_STATUS, font_style)
        ws.write(row_num, 12, my_row.Prediction, font_style)

    wb.save(response)
    return response

@admin_required
def train_model(request):
    detection_accuracy.objects.all().delete()

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
        ('extra_trees', Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', ExtraTreesClassifier(n_estimators=300, max_depth=14, class_weight='balanced_subsample', random_state=42))
        ])),
        ('random_forest', Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=300, max_depth=14, class_weight='balanced_subsample', random_state=42))
        ])),
        ('gradient_boosting', Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42))
        ])),
        ('logistic', Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=5000, C=0.75, class_weight='balanced', solver='liblinear', random_state=42))
        ])),
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    best_model = None
    best_score = 0.0
    for name, model in models:
        model.fit(X_train, y_train)
        score = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)
        detection_accuracy.objects.create(names=name.replace('_', ' ').title(), ratio=score)
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        best_model = models[0][1]

    predictions = best_model.predict(X_test)
    try:
        probabilities = best_model.predict_proba(X_test)[:, 1]
    except Exception:
        probabilities = [None] * len(predictions)

    test_output = df.loc[X_test.index, [
        'Account_Id', 'Trans_Id', 'Age', 'Followers', 'NAME_CONTRACT_TYPE', 'GENDER',
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'Label'
    ]].copy()
    test_output['Predicted'] = predictions
    test_output['Prediction_Confidence'] = probabilities
    test_output['Correct'] = test_output['Label'] == test_output['Predicted']
    test_output['Error_Type'] = test_output.apply(
        lambda row: 'Correct' if row['Correct'] else ('False Negative' if row['Label'] == 1 else 'False Positive'),
        axis=1
    )
    test_output.to_csv(os.path.join(settings.BASE_DIR, 'Test_Dataset_With_Predictions.csv'), index=False)

    csv_format = os.path.join(settings.BASE_DIR, 'Results.csv')
    df.to_csv(csv_format, index=False)

    obj = detection_accuracy.objects.all()
    return render(request, 'SProvider/train_model.html', {'objs': obj, 'best_model': best_model})