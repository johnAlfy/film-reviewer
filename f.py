import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.model_selection  import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from random import shuffle
import string
import os
from sklearn.linear_model import LogisticRegression


def readData(dir,type):
    allFiles = []
    files = os.listdir(dir)
    intType=0
    if type=="pos":
        intType=1
    for i in files:
        f = open(dir+i)
        s = f.read()
        s=s.translate(str.maketrans('', '', string.punctuation))
        pair = (s.lower(),intType)
        allFiles.append(pair)

       # print(allFiles[len(allFiles)-1])
    return allFiles

posDir = "D:\\assignment2NLP\\txt_sentoken\\pos\\"
negDir = "D:\\assignment2NLP\\txt_sentoken\\neg\\"
posList = readData(posDir,"pos")
negList = readData(negDir,"neg")
generalList=negList+posList
shuffle(generalList)

df_x=[]
df_y=[]
for i in range(len(generalList)):
    df_x.append(generalList[i][0])
    df_y.append(generalList[i][1])
tfIdf = TfidfVectorizer(min_df=1,stop_words='english')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train_tfIdf = tfIdf.fit_transform(x_train)
x_test_tfIdf = tfIdf.transform(x_test)

mnb = LogisticRegression()#MultinomialNB()
mnb.fit(x_train_tfIdf, y_train)
predictions = mnb.predict(x_test_tfIdf)
count = 0
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        count = count+1
print(count/len(predictions))

userInput="input.txt"
file=open(userInput)
text=file.read()
text=text.lower()
text_tfIdf=tfIdf.transform([text])
textPrediction=mnb.predict(text_tfIdf)
if(textPrediction[0]==1):
    print("positive review")
else:
    print("negative review")
chi2score = chi2(x_train_tfIdf, y_train)[0]
plt.figure(figsize=(15,10))
wscores = zip(tfIdf.get_feature_names(), chi2score)
wchi2 = sorted(wscores, key=lambda x:x[1])
topchi2 = list(zip(*wchi2[-50:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x,topchi2[1], align='center', alpha=0.2)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')
plt.show()



from sklearn.decomposition import TruncatedSVD
pca = TruncatedSVD(n_components=2)
proj = pca.fit_transform(x_test_tfIdf)
for i in range(len(y_test)):
    if y_test[i] == 0:
        plt.scatter(proj[:,0][i], proj[:,1][i],c='red')
    else:
        plt.scatter(proj[:,0][i], proj[:,1][i], c='green')
plt.show()
print(len(y_test))
# plt.colorbar()
