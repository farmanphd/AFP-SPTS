

# scikit-learn :
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
#from neupy import algorithms
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
#from neupy.algorithms import GRNN as grnn
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier,  AdaBoostClassifier,    GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, \
    RandomForestClassifier, \
    AdaBoostClassifier, \
    GradientBoostingClassifier, \
    ExtraTreesClassifier

Names = ['MLP','RF', 'DT','EXT','Adaboost']

Classifiers = [

  
    #XGBClassifier(n_estimators=200),
    #MLPClassifier(hidden_layer_sizes=15, activation='relu', solver='adam'),
    #RandomForestClassifier(n_estimators=200),
    #DecisionTreeClassifier(),
    #ExtraTreesClassifier(n_estimators=200),
    #AdaBoostClassifier(n_estimators=200)

    #  MLPClassifier(),
  #  RandomForestClassifier(),
   # DecisionTreeClassifier(),
   # ExtraTreesClassifier(),
   # AdaBoostClassifier()
#XGBClassifier(),
   
    
]


def runClassifiers():
    i=0
    Results = []  # compare algorithms
    #cv = StratifiedKFold(n_splits=8, shuffle=True)
    from sklearn.model_selection import StratifiedKFold,KFold
    
    for classifier, name in zip(Classifiers, Names):
        accuray = []
        auROC= []
        avePrecision = []
        F1_Score = []
        AUC = []
        MCC = []
        Recall = []
        mean_TPR = 0.0
        mean_FPR = np.linspace(0, 1, 100)
        CM = np.array([
            [0, 0],
            [0, 0],
        ], dtype=int)
        print(classifier.__class__.__name__)
        model = classifier
        #counter=41
        #counter = 51
        yProCounter = 0
        yThreshCounter = 0
        yDecisionfunCounter = 0
        for (train_index, test_index) in cv.split(X, y):
            X_train = X[train_index]

            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            model.fit(X_train, y_train)
            # Calculate ROC Curve and Area the Curve
            y_artificial = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            FPR, TPR, _ = roc_curve(y_test, y_proba)
            mean_TPR += np.interp(mean_FPR, FPR, TPR)
            mean_TPR[0] = 0.0
            roc_auc = auc(FPR, TPR)
            CM = confusion_matrix(y_true=y_test, y_pred=y_artificial)
            TN, FP, FN, TP = CM.ravel()
            
           


'''def boxPlot(Results, Names):
    ### Algoritms Comparison ###
    # boxplot algorithm comparison
    fig = plt.figure()
    # fig.suptitle('Classifier Comparison')
    ax = fig.add_subplot(111)
    ax.yaxis.grid(True)
    plt.boxplot(Results, patch_artist=True, vert = True, whis=True, showbox=True)
    ax.set_xticklabels(Names)
    plt.xlabel('\nName of Classifiers')
    plt.ylabel('\nAccuracy (%)')

    plt.savefig('AccuracyBoxPlot.png', dpi=100)
    plt.show()'''
    ### --- ###
def auROCplot():
    ### auROC ###
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random')
    plt.xlim([0.0, 1.00])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    # plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig('auROC.png', dpi=300)
    plt.show()
    ### --- ###
if __name__ == '__main__':
    runClassifiers()
   # auROCplot()


