import pandas as pd # Import pandas
import numpy as np # Import numpy
from sklearn.preprocessing import scale # Import scale function (equivalent to scale() in R)
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix # Import confusion matrix
from sklearn.metrics import average_precision_score # ap() in R


pd.set_option("display.max_rows", None, "display.max_columns", None) # Show full dataframe including columns and rows

# This function is to calculate average precision
def ap(actual, predicted):
    sum = 0
    hit = 0

    # Create a for loop 
    for i in range(1, len(predicted)):
        if ((predicted[i] == actual[i]) & (actual[i] == 'useful')):
            hit += 1
            prec = hit/i
            sum += prec

    return (sum/hit) # Return average precision

mydata = pd.read_table(r'/Users/macbook/Desktop/Kim Heejun Review Evaluation Project/Data/yelp_review_features_even.txt', sep='\t')

# See information of the dataframe (equivalent to str(mydata) in R)
# Set verbose=False to see the short summary 
print(mydata.info(verbose=True)) 

# See full statistical summary (equivalent to summary(mydata) in R)
print(mydata.describe()) 

# See all columns (equivalent to names(mydata) in R)
print(mydata.columns.values)

# Indexing dataframe
mydata = pd.concat([mydata.iloc[:, 3:144], mydata.iloc[:, 148]], axis=1)

# Remove component that is not correct values

column_names = list(mydata) # List of headers (columns) in mydata
to_numeric_columns = [] # List of columns that will be converted to numeric

to_numeric_columns += column_names[0:103] 
to_numeric_columns += column_names[104:120] 
to_numeric_columns += column_names[121:126]
to_numeric_columns += column_names[127:136] 
to_numeric_columns += column_names[137:141]

# Use pd.to_numeric()
mydata[to_numeric_columns] = mydata[to_numeric_columns].apply(pd.to_numeric, errors='coerce')

scaled_mydata = pd.DataFrame(scale(mydata[to_numeric_columns]), index=mydata[to_numeric_columns].index, columns=mydata[to_numeric_columns].columns)
scaled_mydata['open'] = mydata['open'] # equivalent to scaled.mydata$open <- mydata$open
scaled_mydata['elite_2015'] = mydata['elite_2015']
scaled_mydata['class'] = mydata['useful']

del mydata # free up some RAM

# Remove columns which have all 'NaN' in their column 
scaled_column_names = list(scaled_mydata) # List of headers (columns) in scaled_mydata
scaled_mydata = scaled_mydata.replace({'0.0':np.nan, 0.0:np.nan}) # Replace '0.0' values with NaN
columns_without_na = [col for col in scaled_column_names if scaled_mydata[col].isna().sum() != len(scaled_mydata.index)] # List of good columns

new_data = scaled_mydata[columns_without_na] # New dataset with good columns only
# These two columns should contain 0 only instead of NaN
new_data.loc[:,'open'] = new_data.loc[:,'open'].fillna(0) 
new_data.loc[:,'elite_2015'] = new_data.loc[:,'elite_2015'].fillna(0)

print(new_data.info())
# Remove votes_funny, useful, cool, sum as they are self-indicative for usefulness
# And make a new order.
new_columns = [
    new_data.iloc[:, :68], new_data.iloc[:, 107], new_data.iloc[:, 72:100], 
    new_data.iloc[:, 106], new_data.iloc[:, 100:106], new_data.iloc[:, 108]
    ]
new_data = pd.concat(new_columns, axis=1) 

n = len(new_data['class']) # Get length of 'class' column
print(n) # See n

folds = 30 # Set number of folds
print(folds) # See folds

# Set random variable reproduciable for desire for reproducible results.
# The number is a reference to a famous book. Same number will keep the random numbers as same.
np.random.seed(107)
list_of_choices = list(range(1, folds+1))
new_data['group'] = np.random.choice(list_of_choices, n, replace=True) # equivalent to sample(1:folds, n, replace=True) in R

# check that we get mean of 0 and sd of 1
# colMeans(scaled.mydata)  # faster version of apply(scaled.dat, 2, mean) in R
# apply(scaled.mydata, 2, sd) in R

# Set business variable
business = ["business_stars", "business_review_count", "open"]

# Set review variable
review_informational_info = ["word_count", "price_included", "procon_included", "stars_included"]
review_informational_star = ["review_stars", "diff_review_user_stars", "diff_review_business_stars"]

review_informational = review_informational_info + review_informational_star
review_readability = [
    "lexical_diversity", "correct_spell_ratio", "ARI", "FleschReadingEase",
    "FleschKincaidGradeLevel", "GunningFogIndex", "SMOGIndex", "ColemanLiauIndex", 
    "LIX", "RIX"
    ]

# These are when usefulness > 5
#review_sentiment = [
#   "positive_emotion", "joy", "love", "affection", "liking", "enthusiasm", 
#   "gratitude", "calmness", "fearlessness", "gladness", "not_sadness", "not_shame",
#   "not_compassion", "not_horror", "not_panic", "not_scare",
#   "not_timidity", "not_distress", "negative_emotion", "sadness", "shame",
#   "compassion", "humility", "despair", "daze", "alarm", "horror", "hysteria", "panic",
#   "scare", "apprehension", "timidity", "distress", "not_love", "not_affection", 
#   "not_enthusiasm", "not_gratitude","neutral_emotion", "apathy", "polarity", "subjectivity", #61
#   "positive_emotion_times_business_stars", "negative_emotion_times_business_stars",
#   "positive_emotion_times_review_stars", "negative_emotion_times_review_stars"
#   ] #65

review_sentiment = [
    "positive_emotion", "joy", "love", "affection", "liking", "enthusiasm", 
    "gratitude", "levity", "calmness", "fearlessness", "gladness", "not_sadness", "not_shame",
    "not_compassion", "not_despair", "not_daze", "not_horror", "not_panic", "not_scare",
    "not_apprehension", "not_timidity", "not_distress", "negative_emotion", "sadness", "shame",
    "compassion", "humility", "despair", "daze", "alarm", "horror", "hysteria", "panic",
    "scare", "apprehension", "timidity", "distress", "not_joy", "not_love", "not_affection", "not_liking",
    "not_enthusiasm", "not_gratitude", "not_gladness",  "neutral_emotion", "apathy", "polarity", "subjectivity",
    "positive_emotion_times_business_stars", "negative_emotion_times_business_stars",
    "positive_emotion_times_review_stars", "negative_emotion_times_review_stars"
    ] 

review = review_informational + review_readability + review_sentiment

# Set reviewer variable
reviewer_reputation_extrinsic = [
    "user_review_count", "elite_2015", "review_count_per_yelping_months", "fans_per_yelping_months", 
    "average_stars", "yelping_months",  "compliments_profile_per_yelping_months",
    "compliments_funny_per_yelping_months", "compliments_cute_per_yelping_months", "compliments_plain_per_yelping_months",
    "compliments_writer_per_yelping_months", "compliments_list_per_yelping_months", "compliments_note_per_yelping_months",
    "compliments_photos_per_yelping_months", "compliments_hot_per_yelping_months", "compliments_cool_per_yelping_months",
    "compliments_more_per_yelping_months", "compliments_sum_per_yelping_months" 
    ]

reviewer_reputation_intrinsic = [
    "degree_within", "betweenness_within", 
    "eigenvector_within", "clusteringCoefficient_within", "closeness_within", "degree_all",
    "betweenness_all", "eigenvector_all", "clusteringCoefficient_all", "closeness_all" 
    ]

reviewer_reputation = reviewer_reputation_extrinsic + reviewer_reputation_intrinsic

reviewer_mobility = ['total_distance_to_centroid', 'navigation_distance', "aggregate_distance", "num_visits"]

reviewer = reviewer_mobility + reviewer_reputation

accuracy_all_all = []
precision_all_all = []
recall_all_all = []
averageprecision_all_all = []
validation_with_error = 0

new_data = new_data.rename(columns={'class':"Class"}) # Keeping 'class' as a column name will raise a syntax error when run GLM

# 30-fold cross-validation 
for i in range(1, folds+1):
    testing = new_data[new_data['group'] == i] # Creating testing set based on values of 'group' column
    training = new_data[new_data['group'] != i] # Training set will be the rest of the dataset

    # Since Python glm formula sees "Class ~ ." as an error, we have to create a string that contains the formula like below
    dependent_vars = ''

    for var in training.columns.values:
        if var != 'Class':
            dependent_vars += var
            dependent_vars += ' + '

    dependent_vars = dependent_vars[:-3] # Drop the last " + " sign

    formula = 'Class ~ ' + dependent_vars # This is the formula

    modelFit = sm.formula.glm(formula=formula, data=training, family=sm.families.Binomial()).fit()

    # print(mylogit.summary()) to see the summary of the model
    modelPred = modelFit.predict(testing)

    # print(modelPred). Don't know why it didn't show probalities, but 1 and 0.
    # 0 = not_useful 
    # 1 = useful

    modelPred = modelPred.map({0.0: 'not_useful', 1.0: 'useful'}) # Change 0 and 1 to 'not_useful' and 'useful'
    
    if modelPred.isnull().values.any() == False:
        # Calculate confusion matrix and other metric scores
        con_matrix = confusion_matrix(testing['Class'], modelPred) # Confusion matrix
        accuracy = (con_matrix[0,0] + con_matrix[1,1]) / np.sum(con_matrix) # Accuracy = (TP+TN)/(TP+FP+TN+FN)
        precision = con_matrix[0,0] / (con_matrix[0,0] + con_matrix[1,0]) # Precision = TP/(TP+FP)
        recall = con_matrix[0,0] / (con_matrix[0,0] + con_matrix[0,1]) # Recall = TP/(TP+FN)
        
        # pos_label inside average_precision_score() only takes binary label of y_true.
        modelPred = modelPred.map({'not_useful':0, 'useful':1})
        testing.loc[:, 'Class'] = testing['Class'].map({'not_useful':0, 'useful':1})
        averageprecision = average_precision_score(testing['Class'], modelPred, pos_label=1) # Average precision score

        # Append results to appropriate lists
        accuracy_all_all.append(accuracy)
        precision_all_all.append(precision)
        recall_all_all.append(recall)
        averageprecision_all_all.append(averageprecision)
    else:
        validation_with_error += 1

# validation_with_error = 5

num_features = 9

accuracy_all_without_review_informational_info = []
precision_all_without_review_informational_info = []
recall_all_without_review_informational_info = []
averageprecision_all_without_review_informational_info = []

accuracy_all_without_review_informational_star = []
precision_all_without_review_informational_star = []
recall_all_without_review_informational_star = []
averageprecision_all_without_review_informational_star = []

accuracy_all_without_review_informational = []
precision_all_without_review_informational = []
recall_all_without_review_informational = []
averageprecision_all_without_review_informational = []

accuracy_all_without_review_readability = []
precision_all_without_review_readability = []
recall_all_without_review_readability = []
averageprecision_all_without_review_readability = []

accurary_all_without_review_sentiment = []
precision_all_without_review_sentiment = []
recall_all_without_review_sentiment = []
averageprecision_all_without_review_sentiment = []

accuracy_all_without_review = []
precision_all_without_review = []
recall_all_without_review = []
averageprecision_all_without_review = []

accuracy_all_without_reviewer_reputation_extrinsic = []
precision_all_without_reviewer_reputation_extrinsic = []
recall_all_without_reviewer_reputation_extrinsic = []
averageprecision_all_without_reviewer_reputation_extrinsic = []

accuracy_all_without_reviewer_reputation_intrinsic = []
precision_all_without_reviewer_reputation_intrinsic = []
recall_all_without_reviewer_reputation_intrinsic = []
averageprecision_all_without_reviewer_reputation_intrinsic = []

accuracy_all_without_reviewer_reputation = []
precision_all_without_reviewer_reputation = []
recall_all_without_reviewer_reputation = []
averageprecision_all_without_reviewer_reputation = []

# Ablation test (not leave-one-out method). We're gonna opt out 1 set of data at a time and do 30-fold cross-validation on the rest.
for i in range(1, num_features):
    if i == 1:
        new_data_temp = new_data.loc[:, [col for col in new_data.columns.values if col not in review_informational_info]]
    elif i == 2:
        new_data_temp = new_data.loc[:, [col for col in new_data.columns.values if col not in review_informational_star]]
    elif i == 3:
        new_data_temp = new_data.loc[:, [col for col in new_data.columns.values if col not in review_informational]]
    elif i == 4:
        new_data_temp = new_data.loc[:, [col for col in new_data.columns.values if col not in review_readability]]
    elif i == 5:
        new_data_temp = new_data.loc[:, [col for col in new_data.columns.values if col not in review_sentiment]]
    elif i == 6:
        new_data_temp = new_data.loc[:, [col for col in new_data.columns.values if col not in review]]
    elif i == 7:
        new_data_temp = new_data.loc[:, [col for col in new_data.columns.values if col not in reviewer_reputation_extrinsic]]
    elif i == 8:
        new_data_temp = new_data.loc[:, [col for col in new_data.columns.values if col not in reviewer_reputation_intrinsic]]
    elif i == 9:
        new_data_temp = new_data.loc[:, [col for col in new_data.columns.values if col not in reviewer_reputation]]
    
    # 30-fold cross-validation again
    for j in range(1, folds+1):
        testing = new_data_temp[new_data_temp['group'] == i]
        training = new_data_temp[new_data_temp['group'] != i] 

        dependent_vars = ''

        for var in training.columns.values:
            if var != 'Class':
                dependent_vars += var
                dependent_vars += ' + '

        dependent_vars = dependent_vars[:-3] 

        formula = 'Class ~ ' + dependent_vars 

        modelFit = sm.formula.glm(formula=formula, data=training, family=sm.families.Binomial()).fit()

        # print(mylogit.summary()) to see the summary of the model
        modelPred = modelFit.predict(testing)
        modelPred = modelPred.map({0.0: 'not_useful', 1.0: 'useful'}) 

        if modelPred.isnull().values.any() == False:
            # Calculate confusion matrix and other metric scores
            con_matrix = confusion_matrix( testing['Class'], modelPred) # Confusion matrix
            accuracy = (con_matrix[0,0] + con_matrix[1,1]) / np.sum(con_matrix) # Accuracy = (TP+TN)/(TP+FP+TN+FN)
            precision = con_matrix[0,0] / (con_matrix[0,0] + con_matrix[1,0]) # Precision = TP/(TP+FP)
            recall = con_matrix[0,0] / (con_matrix[0,0] + con_matrix[0,1]) # Recall = TP/(TP+FN)
            
            modelPred = modelPred.map({'not_useful':0, 'useful':1})
            testing.loc[:, 'Class'] = testing['Class'].map({'not_useful':0, 'useful':1})
            averageprecision = average_precision_score(testing['Class'], modelPred, pos_label=1) # Average precision score

            # Append results to appropriate lists
            if i == 1:  
                accuracy_all_without_review_informational_info.append(accuracy)
                precision_all_without_review_informational_info.append(precision)
                recall_all_without_review_informational_info.append(recall)
                averageprecision_all_without_review_informational_info.append(averageprecision)        
            elif i == 2:
                accuracy_all_without_review_informational_star.append(accuracy)
                precision_all_without_review_informational_star.append(precision)
                recall_all_without_review_informational_star.append(recall)
                averageprecision_all_without_review_informational_star.append(averageprecision)        
            elif i == 3:
                accuracy_all_without_review_informational.append(accuracy)
                precision_all_without_review_informational.append(precision)
                recall_all_without_review_informational.append(recall)
                averageprecision_all_without_review_informational.append(averageprecision)        
            elif i == 4:
                accuracy_all_without_review_readability.append(accuracy)
                precision_all_without_review_readability.append(precision)
                recall_all_without_review_readability.append(recall)
                averageprecision_all_without_review_readability.append(averageprecision)  
            elif i == 5:
                accurary_all_without_review_sentiment.append(accuracy)
                precision_all_without_review_sentiment.append(precision)
                recall_all_without_review_sentiment.append(recall)
                averageprecision_all_without_review_sentiment.append(averageprecision)  
            elif  i == 6:
                accuracy_all_without_review.append(accuracy)
                precision_all_without_review.append(precision)
                recall_all_without_review.append(recall)
                averageprecision_all_without_review.append(averageprecision)  
            elif i == 7:
                accuracy_all_without_reviewer_reputation_extrinsic.append(accuracy)
                precision_all_without_reviewer_reputation_extrinsic.append(precision)
                recall_all_without_reviewer_reputation_extrinsic.append(recall)
                averageprecision_all_without_reviewer_reputation_extrinsic.append(averageprecision)  
            elif i == 8:
                accuracy_all_without_reviewer_reputation_intrinsic.append(accuracy)
                precision_all_without_reviewer_reputation_intrinsic.append(precision)
                recall_all_without_reviewer_reputation_intrinsic.append(recall)
                averageprecision_all_without_reviewer_reputation_intrinsic.append(averageprecision)  
            elif i == 9:
                accuracy_all_without_reviewer_reputation.append(accuracy)
                precision_all_without_reviewer_reputation.append(precision)
                recall_all_without_reviewer_reputation.append(recall)
                averageprecision_all_without_reviewer_reputation.append(averageprecision) 

# Create series of score values and their headers
score1 = pd.Series(accuracy_all_all, name='accuracy_all_all')
score2 = pd.Series(recall_all_all, name='recall_all_all')
score3 = pd.Series(precision_all_all, name='precision_all_all')
score4 = pd.Series(averageprecision_all_all, name='averageprecision_all_all')
score5 = pd.Series(accuracy_all_without_review_informational_info, name='accuracy_all_without_review_informational_info')
score6 = pd.Series(precision_all_without_review_informational_info, name='precision_all_without_review_informational_info')
score7 = pd.Series(recall_all_without_review_informational_info, name='recall_all_without_review_informational_info')
score8 = pd.Series(averageprecision_all_without_review_informational_info, name='averageprecision_all_without_review_informational_info')
score9 = pd.Series(accuracy_all_without_review_informational_star, name='accuracy_all_without_review_informational_star')
score10 = pd.Series(precision_all_without_review_informational_star, name='precision_all_without_review_informational_star')
score11 = pd.Series(recall_all_without_review_informational_star, name='recall_all_without_review_informational_star')
score12 = pd.Series(averageprecision_all_without_review_informational_star, name='averageprecision_all_without_review_informational_star')
score13 = pd.Series(accuracy_all_without_review_informational, name='accuracy_all_without_review_informational')
score14 = pd.Series(precision_all_without_review_informational, name='precision_all_without_review_informational')
score15 = pd.Series(recall_all_without_review_informational, name='recall_all_without_review_informational')
score16 = pd.Series(averageprecision_all_without_review_informational, name='averageprecision_all_without_review_informational')
score17 = pd.Series(accuracy_all_without_review_readability, name='accuracy_all_without_review_readability')
score18 = pd.Series(precision_all_without_review_readability, name='precision_all_without_review_readability')
score19 = pd.Series(recall_all_without_review_readability, name='recall_all_without_review_readability')
score20 = pd.Series(averageprecision_all_without_review_readability, name='averageprecision_all_without_review_readability')
score21 = pd.Series(accurary_all_without_review_sentiment, name='accurary_all_without_review_sentiment')
score22= pd.Series(precision_all_without_review_sentiment, name='precision_all_without_review_sentiment')
score23 = pd.Series(recall_all_without_review_sentiment, name='recall_all_without_review_sentiment')
score24 = pd.Series(averageprecision_all_without_review_sentiment, name='averageprecision_all_without_review_sentiment')
score25 = pd.Series(accuracy_all_without_review, name='accuracy_all_without_review')
score26 = pd.Series(precision_all_without_review, name='precision_all_without_review')
score27 = pd.Series(recall_all_without_review, name='recall_all_without_review')
score28 = pd.Series(averageprecision_all_without_review, name='averageprecision_all_without_review')
score29 = pd.Series(accuracy_all_without_reviewer_reputation_extrinsic, name='accuracy_all_without_reviewer_reputation_extrinsic')
score30 = pd.Series(precision_all_without_reviewer_reputation_extrinsic, name='precision_all_without_reviewer_reputation_extrinsic')
score31 = pd.Series(recall_all_without_reviewer_reputation_extrinsic, name='recall_all_without_reviewer_reputation_extrinsic')
score32 = pd.Series(averageprecision_all_without_reviewer_reputation_extrinsic, name='averageprecision_all_without_reviewer_reputation_extrinsic')
score33 = pd.Series(accuracy_all_without_reviewer_reputation_intrinsic, name='accuracy_all_without_reviewer_reputation_intrinsic')
score34 = pd.Series(precision_all_without_reviewer_reputation_intrinsic, name='precision_all_without_reviewer_reputation_intrinsic')
score35 = pd.Series(recall_all_without_reviewer_reputation_intrinsic, name='recall_all_without_reviewer_reputation_intrinsic')
score36 = pd.Series(averageprecision_all_without_reviewer_reputation_intrinsic, name='averageprecision_all_without_reviewer_reputation_intrinsic')
score37 = pd.Series(accuracy_all_without_reviewer_reputation, name='accuracy_all_without_reviewer_reputation')
score38 = pd.Series(precision_all_without_reviewer_reputation, name='precision_all_without_reviewer_reputation')
score39 = pd.Series(recall_all_without_reviewer_reputation, name='recall_all_without_reviewer_reputation')
score40 = pd.Series(averageprecision_all_without_reviewer_reputation, name='averageprecision_all_without_reviewer_reputation')


# Concat all the above series into a dataframe. 
# Have to do this way instead of creating a dictionary then convert to dataframe because the scores lists don't have the same length. 
# Concatenating series will fill in NaN. 
results = pd.concat([
    score1, score2, score3, score4, score5, score6, score7, score8, score9, score10,
    score11, score12, score13, score14, score15, score16, score17, score18, score19, score 20,
    score21, score22, score23, score24, score25, score26, score27, score28, score29, score 30,
    score31, score32, score33, score34, score35, score36, score37, score38, score39, score 40
], axis=1)

# Export to a text file
results.to_csv(r'/Users/macbook/Desktop/Kim Heejun Review Evaluation Project/Yelp_Usefulness_Results_sub_category_with_ap.txt', index=None, sep=' ', mode='a')

# Export to a csv file
results.to_csv(r'/Users/macbook/Desktop/Kim Heejun Review Evaluation Project/Yelp_Usefulness_Results_sub_category_with_ap.csv')