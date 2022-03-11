import keras
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,classification_report

model=keras.models.load_model('/CASIA 2.0/CasiaAttention_BestModel4.h5')

#Collecting data for different sets according to 5-fold cross validation
X_train1,X_train2,X_train3,X_train4,X_train5=data[train_lt[0]],data[train_lt[1]],data[train_lt[2]],data[train_lt[3]],data[train_lt[4]]
X_test1,X_test2,X_test3,X_test4,X_test5=data[test_lt[0]],data[test_lt[1]],data[test_lt[2]],data[test_lt[3]],data[test_lt[4]]

y_train1,y_train2,y_train3,y_train4,y_train5=target[train_lt[0]],target[train_lt[1]],target[train_lt[2]],target[train_lt[3]],target[train_lt[4]]
y_test1,y_test2,y_test3,y_test4,y_test5=target[test_lt[0]],target[test_lt[1]],target[test_lt[2]],target[test_lt[3]],target[test_lt[4]]

#Predicting output for all sets
y_train_pred1=model.predict(X_train1)
y_test_pred1=model.predict(X_test1)
y_train_pred2=model.predict(X_train2)
y_test_pred2=model.predict(X_test2)
y_train_pred3=model.predict(X_train3)
y_test_pred3=model.predict(X_test3)
y_train_pred4=model.predict(X_train4)
y_test_pred4=model.predict(X_test4)
y_train_pred5=model.predict(X_train5)
y_test_pred5=model.predict(X_test5)

y_train_pred1=np.argmax(y_train_pred1,axis=-1)
y_train_label1=np.argmax(y_train1,axis=-1)
y_test_pred1=np.argmax(y_test_pred1,axis=-1)
y_test_label1=np.argmax(y_test1,axis=-1)

y_train_pred2=np.argmax(y_train_pred2,axis=-1)
y_train_label2=np.argmax(y_train2,axis=-1)
y_test_pred2=np.argmax(y_test_pred2,axis=-1)
y_test_label2=np.argmax(y_test2,axis=-1)

y_train_pred3=np.argmax(y_train_pred3,axis=-1)
y_train_label3=np.argmax(y_train3,axis=-1)
y_test_pred3=np.argmax(y_test_pred3,axis=-1)
y_test_label3=np.argmax(y_test3,axis=-1)

y_train_pred4=np.argmax(y_train_pred4,axis=-1)
y_train_label4=np.argmax(y_train4,axis=-1)
y_test_pred4=np.argmax(y_test_pred4,axis=-1)
y_test_label4=np.argmax(y_test4,axis=-1)

y_train_pred5=np.argmax(y_train_pred5,axis=-1)
y_train_label5=np.argmax(y_train5,axis=-1)
y_test_pred5=np.argmax(y_test_pred5,axis=-1)
y_test_label5=np.argmax(y_test5,axis=-1)


train_acc1=accuracy_score(y_train_pred1,y_train_label1)
test_acc1=accuracy_score(y_test_pred1,y_test_label1)

train_acc2=accuracy_score(y_train_pred2,y_train_label2)
test_acc2=accuracy_score(y_test_pred2,y_test_label2)

train_acc3=accuracy_score(y_train_pred3,y_train_label3)
test_acc3=accuracy_score(y_test_pred3,y_test_label3)

train_acc4=accuracy_score(y_train_pred4,y_train_label4)
test_acc4=accuracy_score(y_test_pred4,y_test_label4)

train_acc5=accuracy_score(y_train_pred5,y_train_label5)
test_acc5=accuracy_score(y_test_pred5,y_test_label5)


# Printing the Classification Matrix
final_train_acc=sum([train_acc1,train_acc2,train_acc3,train_acc4,train_acc5])/5
final_test_acc=sum([test_acc1,test_acc2,test_acc3,test_acc4,test_acc5])/5
final_train_acc,final_test_acc


report=classification_report(y_test_label4,y_test_pred4)
print(report)

# Printing the Confusion Matrix

cm=confusion_matrix(y_test_label4,y_test_pred4)
print(cm)

plt.figure(figsize = (8,4))
sn.heatmap(cm, annot=True,fmt='d',cmap='Blues')
plt.title('confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# Printing the ROC Curve

from sklearn.metrics import roc_curve,roc_auc_score,auc
train_fpr, train_tpr, train_threshold = roc_curve(y_train_label4, y_train_pred4)
train_AUC = auc(train_fpr, train_tpr)
train_AUC

fpr, tpr, threshold = roc_curve(y_test_label4, y_test_pred4)
test_AUC = auc(fpr, tpr)
test_AUC

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % test_AUC)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()