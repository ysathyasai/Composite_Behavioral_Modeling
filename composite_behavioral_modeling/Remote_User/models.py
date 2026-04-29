from django.db import models


# -------------------------------
# Client Registration Model
# -------------------------------
class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=100)

    # NOTE: Still plain text (not secure) – fix later using Django auth
    password = models.CharField(max_length=128)

    phoneno = models.CharField(max_length=15)

    country = models.CharField(max_length=50)
    state = models.CharField(max_length=50)
    city = models.CharField(max_length=50)

    gender = models.CharField(max_length=10)

    # FIXED: No length limit issue anymore
    address = models.TextField()

    def __str__(self):
        return self.username


# -------------------------------
# Identity Theft Detection Model
# -------------------------------
class identity_theft_detection(models.Model):

    Account_Id = models.CharField(max_length=100)
    Trans_Id = models.CharField(max_length=100)

    # FIXED: Proper data types
    Age = models.IntegerField()
    Followers = models.IntegerField()

    NAME_CONTRACT_TYPE = models.CharField(max_length=100)
    GENDER = models.CharField(max_length=10)

    AMT_INCOME_TOTAL = models.FloatField()
    AMT_CREDIT = models.FloatField()
    AMT_ANNUITY = models.FloatField()
    AMT_GOODS_PRICE = models.FloatField()

    NAME_INCOME_TYPE = models.CharField(max_length=100)
    NAME_FAMILY_STATUS = models.CharField(max_length=100)

    Prediction = models.CharField(max_length=50)

    def __str__(self):
        return self.Account_Id


# -------------------------------
# Detection Accuracy Model
# -------------------------------
class detection_accuracy(models.Model):
    names = models.CharField(max_length=100)
    ratio = models.FloatField()

    def __str__(self):
        return self.names


# -------------------------------
# Detection Ratio Model
# -------------------------------
class detection_ratio(models.Model):
    names = models.CharField(max_length=100)
    ratio = models.FloatField()

    def __str__(self):
        return self.names
