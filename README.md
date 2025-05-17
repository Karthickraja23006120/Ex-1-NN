<H3>ENTER YOUR NAME: Karthick Raja K</H3>
<H3>ENTER YOUR REGISTER NO: 212223240066</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))


```


## OUTPUT:
### Dataset:
![Screenshot 2025-05-17 123151](https://github.com/user-attachments/assets/bad46f33-727c-4943-90ec-325012140982)

### X Values:
![Screenshot 2025-05-17 123405](https://github.com/user-attachments/assets/6b50e755-5eab-4291-8e96-5cdcb922f5f0)

### Y Values:
![Screenshot 2025-05-17 123521](https://github.com/user-attachments/assets/be075ef5-55fa-4bbc-967f-a7fe346fb8ca)

### Null Values:
![Screenshot 2025-05-17 123619](https://github.com/user-attachments/assets/7100b755-ec23-4c0e-a955-d7b1e0fe6e97)

### Duplicated Values:
![Screenshot 2025-05-17 123704](https://github.com/user-attachments/assets/640bcbe0-dce5-4158-b417-063cdb931688)

### Description:
![Screenshot 2025-05-17 123752](https://github.com/user-attachments/assets/89ac2067-91d8-4cdd-a3a7-4725f7711efb)

### Normalized Dataset:
![Screenshot 2025-05-17 124038](https://github.com/user-attachments/assets/9dbdc5b8-f9af-43c3-a43e-0cb5d594e91f)

### Training Data:
![Screenshot 2025-05-17 124547](https://github.com/user-attachments/assets/80eeae26-1f47-4c1d-88ec-a7c82e641757)


### Testing Data:
![Screenshot 2025-05-17 124614](https://github.com/user-attachments/assets/7311db74-8e80-4957-80e1-7c6a797a30b0)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


