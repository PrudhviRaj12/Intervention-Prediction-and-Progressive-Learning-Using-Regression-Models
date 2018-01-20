
# coding: utf-8

# In[10]:

import graphlab
students = graphlab.SFrame("Students Dataset Final.csv")

def simple_linear_regression(input_feature, output):
    sum_input_feature = input_feature.sum()
    sum_output = output.sum()
    size_input_feature = input_feature.size()
    size_output = output.size()
    mean_input_feature = sum_input_feature/size_input_feature
    mean_output = sum_output/size_output
    
    prod_input_feature_output = ((input_feature * output)).sum()
    mean_prod_input_feature_output = (sum_input_feature * sum_output)/size_input_feature
    
    squared_input_feature = (input_feature ** 2).sum()
    mean_prod_input_feature_twice = (sum_input_feature * sum_input_feature)/size_input_feature
    
    slope_numerator = prod_input_feature_output - mean_prod_input_feature_output
    slope_denominator = squared_input_feature - mean_prod_input_feature_twice
    
    slope = slope_numerator/slope_denominator
    
    intercept = mean_output - (slope * mean_input_feature)
    
    return (intercept, slope)

def regression_predictions(input_feature, intercept, slope):
    predicted_value = intercept + (slope * input_feature)
    return predicted_value

def RSS_closed_form(input_feature, output, intercept, slope):
    predicted_values = regression_predictions(input_feature, intercept, slope)
    residuals = output - predicted_values
    RSS = (residuals ** 2).sum()
    
    return RSS

def RSS_closed_form_two(predicted_value, output):
    residuals = predicted_value - output
    RSS = (residuals ** 2).sum()
    
    return round(RSS, 2)

students['Exclude Quizzes'] = (students['Case Study Average(20)'] + students['Assignment Average(10)'] + students['Mid Average (30)'])
training_data, test_data = students.random_split(0.7, seed=0)

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


quiz_intercept, quiz_slope = simple_linear_regression(training_data['Quiz Average (10)'], training_data['FINAL GRADE'])
training_data['FGP All Quizzes'] = regression_predictions(training_data['Quiz Average (10)'], quiz_intercept, quiz_slope)
RSS_training_data_quiz_average = RSS_closed_form_two(training_data['FGP All Quizzes'], training_data['FINAL GRADE'])
plt.plot(training_data['Quiz Average (10)'],training_data['FINAL GRADE'],'.',
        training_data['Quiz Average (10)'], training_data['FGP All Quizzes'],'-')
test_data['FGP All Quizzes'] = regression_predictions(test_data['Quiz Average (10)'], quiz_intercept, quiz_slope)
RSS_test_data_quiz_average = RSS_closed_form(test_data['FGP All Quizzes'], test_data['FINAL GRADE'], quiz_intercept, quiz_slope)
plt.plot(test_data['Quiz Average (10)'],test_data['FINAL GRADE'],'.',
        test_data['Quiz Average (10)'], test_data['FGP All Quizzes'],'-')

def error_rate (original_output, predicted_output):
    residual = abs(original_output - predicted_output)
    error_rate = (residual/original_output).mean()
    
    return round(error_rate, 3)

training_data_quiz_error_rate = (error_rate(training_data['FINAL GRADE'], training_data['FGP All Quizzes']))
test_data_quiz_error_rate = error_rate(test_data['FINAL GRADE'], test_data['FGP All Quizzes'])
exclude_quiz_intercept, exclude_quiz_slope = simple_linear_regression(training_data['Exclude Quizzes'], training_data['FINAL GRADE'])
training_data['FGP No Quizzes'] = regression_predictions(training_data['Exclude Quizzes'], exclude_quiz_intercept, exclude_quiz_slope)
plt.plot(training_data['Exclude Quizzes'],training_data['FINAL GRADE'],'.',
        training_data['Exclude Quizzes'], training_data['FGP No Quizzes'],'-')

RSS_training_data_no_quiz = RSS_closed_form_two(training_data['FGP No Quizzes'], training_data['FINAL GRADE'])
test_data['FGP No Quizzes'] = regression_predictions(test_data['Exclude Quizzes'], exclude_quiz_intercept, exclude_quiz_slope)
RSS_test_data_no_quiz = RSS_closed_form(test_data['FGP No Quizzes'], test_data['FINAL GRADE'], exclude_quiz_intercept, exclude_quiz_slope)
plt.plot(test_data['Exclude Quizzes'],test_data['FINAL GRADE'],'.',
        test_data['Exclude Quizzes'], test_data['FGP No Quizzes'],'-')
training_data_no_quiz_error_rate = (error_rate(training_data['FINAL GRADE'], training_data['FGP No Quizzes']))
test_data_no_quiz_error_rate = (error_rate(test_data['FINAL GRADE'], test_data['FGP No Quizzes']))

high_error_intercept, high_error_slope = simple_linear_regression(training_data['High-Error Average (10)'], training_data['FINAL GRADE'])
training_data['FGP High-Level Quizzes'] = regression_predictions(training_data['High-Error Average (10)'], high_error_intercept, high_error_slope)
RSS_training_data_high_quiz_average = RSS_closed_form_two(training_data['FGP High-Level Quizzes'], training_data['FINAL GRADE'])
plt.plot(training_data['High-Error Average (10)'],training_data['FINAL GRADE'],'.',
        training_data['High-Error Average (10)'], training_data['FGP High-Level Quizzes'],'-')
test_data['FGP High-Level Quizzes'] = regression_predictions(test_data['High-Error Average (10)'], high_error_intercept, high_error_slope)
RSS_test_data_high_quiz_average = RSS_closed_form(test_data['FGP High-Level Quizzes'], test_data['FINAL GRADE'], high_error_intercept, high_error_slope)
plt.plot(test_data['High-Error Average (10)'],test_data['FINAL GRADE'],'.',
        test_data['High-Error Average (10)'], test_data['FGP High-Level Quizzes'],'-')
training_data_high_quiz_error_rate = (error_rate(training_data['FINAL GRADE'], training_data['FGP High-Level Quizzes']))
test_data_high_quiz_error_rate = (error_rate(test_data['FINAL GRADE'], test_data['FGP High-Level Quizzes']))






#Model 1
training_data['FGP Model Multi 1'] = 0
new_features = ['Quiz Average (10)', 'Mid Average (30)']
model_multi_1 = graphlab.linear_regression.create(training_data, target = 'FINAL GRADE', features= new_features, validation_set= None)
model_multi_1.get('coefficients')
training_data['FGP Model Multi 1'] =model_multi_1.predict(training_data)
training_data_multi_model_1_RSS = RSS_closed_form_two(training_data['FGP Model Multi 1'], training_data['FINAL GRADE'] )
training_data_multi_model_1_error = error_rate(training_data['FINAL GRADE'], training_data['FGP Model Multi 1'] )
test_data['FGP Model Multi 1'] = model_multi_1.predict(test_data)
test_data_multi_model_1_RSS = RSS_closed_form_two(test_data['FGP Model Multi 1'], test_data['FINAL GRADE'] )
test_data_multi_model_1_error = error_rate(test_data['FINAL GRADE'], test_data['FGP Model Multi 1'] )

#Model 2
features_test_2 = ['High-Error Average (10)', 'Mid Average (30)', '*Quiz 11 (10)']
model_multi_2 = graphlab.linear_regression.create(training_data, target = 'FINAL GRADE', features= features_test_2, validation_set= None)
training_data['FGP Model Multi 2'] =model_multi_2.predict(training_data)
training_data_multi_model_2_RSS = RSS_closed_form_two(training_data['FGP Model Multi 2'], training_data['FINAL GRADE'] )
training_data_multi_model_2_error = error_rate(training_data['FINAL GRADE'], training_data['FGP Model Multi 2'] )
test_data['FGP Model Multi 2'] =model_multi_2.predict(test_data)
test_data_multi_model_2_RSS = RSS_closed_form_two(test_data['FGP Model Multi 2'], test_data['FINAL GRADE'] )
test_data_multi_model_2_error = error_rate(test_data['FINAL GRADE'], test_data['FGP Model Multi 2'] )

#Model 3
features_test_3 = ['Case Study Average(20)', 'Mid Average (30)', 'Assignment Average(10)']
model_multi_3 = graphlab.linear_regression.create(training_data, target = 'FINAL GRADE', features= features_test_3, validation_set= None)
model_multi_3.get('coefficients')
training_data['FGP Model Multi 3'] = model_multi_3.predict(training_data)
training_data_multi_model_3_RSS = RSS_closed_form_two(training_data['FINAL GRADE'], training_data['FGP Model Multi 3'])
training_data_multi_model_3_error = error_rate(training_data['FINAL GRADE'], training_data['FGP Model Multi 3'])
test_data['FGP Model Multi 3'] = model_multi_3.predict(test_data)
test_data_multi_model_3_RSS = RSS_closed_form_two(test_data['FINAL GRADE'], test_data['FGP Model Multi 3'])
test_data_multi_model_3_error = error_rate(test_data['FINAL GRADE'], test_data['FGP Model Multi 3'])

#Model 4
model_multi_4_ridge = graphlab.linear_regression.create(training_data, target = 'FINAL GRADE', features= features_test_3, l1_penalty= 0.01e1, max_iterations= 600, validation_set= None)
model_multi_4_ridge.get('coefficients')
training_data['FGP Model Multi 4 Ridge'] = model_multi_4_ridge.predict(training_data)
training_data_multi_model_4_ridge_RSS = RSS_closed_form_two(training_data['FINAL GRADE'], training_data['FGP Model Multi 4 Ridge'])
training_data_multi_model_4_ridge_error = error_rate(training_data['FINAL GRADE'], training_data['FGP Model Multi 4 Ridge'])
test_data['FGP Model Multi 4 Ridge'] = model_multi_4_ridge.predict(test_data)
test_data_multi_model_4_ridge_RSS = RSS_closed_form_two(test_data['FINAL GRADE'], test_data['FGP Model Multi 4 Ridge'])
test_data_multi_model_4_ridge_error = error_rate(test_data['FINAL GRADE'], test_data['FGP Model Multi 4 Ridge'])

#Model 5
all_quizzes = [ '*Quiz 1 (10)','Quiz 2 (10)','Quiz 3 (10)','*Quiz 4 (10)','Quiz 5 (10)', 'Quiz 6 (10)',
               'Quiz 7 (10)','Quiz 8 (10)','Quiz 9 (10)', '*Quiz 10 (10)',  '*Quiz 11 (10)',  'Quiz 12 (10)',
               'Quiz 13  (10)','*Quiz 14 (10)','Quiz 15 (10)', '*Quiz 16 (10)','*Quiz 17 (10)',
               'Quiz 18 (10)', '*Quiz 19 (10)', 'Quiz 20 (10)', 'Quiz 21 (10)', 'Quiz 22 (10)']
len(all_quizzes)
regular_fit_all_quizzes = graphlab.linear_regression.create(training_data, target= "FINAL GRADE", features= all_quizzes,
                                                            validation_set= None)
regular_fit_all_quizzes.get("coefficients")
training_data['Regular All Quizzes Fit'] = regular_fit_all_quizzes.predict(training_data)
training_data_regular_all_quizzes_fit_RSS = RSS_closed_form_two(training_data['FINAL GRADE'], training_data['Regular All Quizzes Fit'])
training_data_regular_all_quizzes_fit_error = error_rate(training_data['FINAL GRADE'], training_data['Regular All Quizzes Fit'])
test_data['Regular All Quizzes Fit'] = regular_fit_all_quizzes.predict(test_data)
test_data_regular_all_quizzes_fit_RSS = RSS_closed_form_two(test_data['FINAL GRADE'], test_data['Regular All Quizzes Fit'])
test_data_regular_all_quizzes_fit_error = error_rate(test_data['FINAL GRADE'], test_data['Regular All Quizzes Fit'])

#Model 6
ridge_fit_all_quizzes = graphlab.linear_regression.create(training_data, target= "FINAL GRADE", features= all_quizzes, 
                                                          l1_penalty= 0.5e1 , max_iterations = 800, validation_set= None)
ridge_fit_all_quizzes.get("coefficients")
training_data['Ridge All Quizzes Fit'] = ridge_fit_all_quizzes.predict(training_data)
training_data_ridge_all_quizzes_fit_RSS = RSS_closed_form_two(training_data['Ridge All Quizzes Fit'], training_data['FINAL GRADE'])
training_data_ridge_all_quizzes_fit_error = error_rate(training_data['FINAL GRADE'], training_data['Ridge All Quizzes Fit'])
test_data['Ridge All Quizzes Fit'] = ridge_fit_all_quizzes.predict(test_data)
test_data_ridge_all_quizzes_fit_RSS = RSS_closed_form_two(test_data['Ridge All Quizzes Fit'], test_data['FINAL GRADE'])

test_data_ridge_all_quizzes_fit_error = error_rate(test_data['FINAL GRADE'], test_data['Ridge All Quizzes Fit'])

#Model 7
mids_and_case_studies = [ 'Mid 1 (30)', 'Mid 2 (30)','Mid 3 (30)', 'Case Study 1 (20)', 'Case Study 2 (20)']
len(mids_and_case_studies)
mids_and_case_studies_fit = graphlab.linear_regression.create(training_data, target = "FINAL GRADE",  
                                                                   features=mids_and_case_studies,
                                                                validation_set = None)
mids_and_case_studies_fit.get('coefficients')
training_data['Mids and Case Studies Fit'] = mids_and_case_studies_fit.predict(training_data)
mids_and_case_studies_fit_train_RSS = RSS_closed_form_two(training_data['Mids and Case Studies Fit'], training_data['FINAL GRADE'])
mids_and_case_studies_fit_train_error = error_rate(training_data['Mids and Case Studies Fit'], training_data['FINAL GRADE'])
test_data['Mids and Case Studies Fit'] = mids_and_case_studies_fit.predict(test_data)
mids_and_case_studies_fit_test_RSS = RSS_closed_form_two(test_data['Mids and Case Studies Fit'], test_data['FINAL GRADE'])
mids_and_case_studies_fit_test_error = error_rate(test_data['Mids and Case Studies Fit'], test_data['FINAL GRADE'])



students['Quiz 1 (10)'] = students['*Quiz 1 (10)']
students['Quiz 4 (10)'] = students['*Quiz 4 (10)']
students['Quiz 10 (10)'] = students['*Quiz 10 (10)']
students['Quiz 11 (10)'] = students['*Quiz 11 (10)']
students['Quiz 13 (10)'] = students['Quiz 13  (10)']
students['Quiz 14 (10)'] = students['*Quiz 14 (10)']
students['Quiz 16 (10)'] = students['*Quiz 16 (10)']
students['Quiz 17 (10)'] = students['*Quiz 17 (10)']
students['Quiz 19 (10)'] = students['*Quiz 19 (10)']

quiz_to_class = {1 : '2', 
                 2 : '3',
                 3 : '4',
                 4 : '5',
                 5 : '6',
                 6 : '7',
                 7 : '9',
                 8 : '10',
                 9 : '11',
                 10: '12',
                 11: '14',
                 12: '15',
                 13: '16',
                 14: '19',
                 15: '20',
                 16: '23',
                 17: '24',
                 18: '25',
                 19: '29',
                 20: '30',
                 21: '31',
                 22: '32'}

def intervention_prediction_on_quizzes_at_every_level(StudentID):
    for i in range(1, len(all_quizzes)-1):
        
        ID = StudentID
        
        if ((students['Quiz ' +str(i) + ' (10)' ][ID-1] + students['Quiz ' +str(i+1) + ' (10)' ][ID-1]+ 
        students['Quiz ' +str(i+2) + ' (10)' ][ID-1])/3) < 5:
            print "Intervention Required as of quiz " + str(i+2)
        
        else:
            print "No Intervention Required as of quiz " + str(i+2)

def intervention_prediction_on_quizzes(StudentID):
    warning = 1
    for i in range(1, len(all_quizzes)-1):
        
        ID = StudentID
        
        if ((students['Quiz ' +str(i) + ' (10)' ][ID-1] + students['Quiz ' +str(i+1) + ' (10)' ][ID-1]+ 
        students['Quiz ' +str(i+2) + ' (10)' ][ID-1])/3) < 5:
            
            if warning < 4:
                print "Intervention Warning " +str(warning) + " quiz " + str(i+2)
                warning = warning + 1
                
            else:
                print "Intervention scheduled after quiz " + str(i+2) + " and after class number " + str(int(int(quiz_to_class[i]) + int(2)))
                    
    if warning == 1:
        print "No Warnings and No Intervention Required"
    elif warning < 4:
        print "No Intervention Required"


#Constructing a list with all the features

total_features= ['*Quiz 1 (10)', 'Quiz 2 (10)', 'Quiz 3 (10)', '*Quiz 4 (10)', 'Quiz 5 (10)', 'Quiz 6 (10)',
              'Assignment 1 (10)', 'Quiz 8 (10)', 'Quiz 9 (10)', 'Case Study 1 (20)', '*Quiz 10 (10)',
               'Mid 1 (30)', '*Quiz 11 (10)', 'Quiz 12 (10)', 'Quiz 13  (10)', 'Assignment 2 (10)', '*Quiz 14 (10)',
              'Quiz 15 (10)', 'Mid 2 (30)', '*Quiz 16 (10)', '*Quiz 17 (10)', 'Quiz 18 (10)', 
               'Case Study 2 (20)', 'Assignment 3 (10)' , '*Quiz 19 (10)', 'Quiz 20 (10)', 
              'Quiz 21 (10)', 'Quiz 22 (10)', 'Mid 3 (30)' ]


# Initializing Lists
model = []
training_data_model_RSS = []
training_data_model_error = []
test_data_model_RSS = []
test_data_model_error = []

#Looping All the features
# features[0:i] is to show the progessive inclusion of features as the course progresses
for i in range(1, len(total_features)+1):
    print i
    model = model + [graphlab.linear_regression.create(training_data, target = 'FINAL GRADE', 
                                                       features= total_features[0:i], validation_set= None)]
    training_data['Model ' +str(i)] = model[i-1].predict(training_data)
    training_data_model_RSS = training_data_model_RSS+ [RSS_closed_form_two(training_data['Model ' +str(i)], 
                                                                            training_data['FINAL GRADE'])]
    training_data_model_error = training_data_model_error + [error_rate(training_data['Model ' +str(i)], 
                                                                        training_data['FINAL GRADE'])]
    test_data['Model ' +str(i)] = model[i-1].predict(test_data)
    test_data_model_RSS = test_data_model_RSS + [RSS_closed_form_two(test_data['Model ' +str(i)], 
                                                                     test_data['FINAL GRADE'])]
    test_data_model_error = test_data_model_error + [error_rate(test_data['Model ' +str(i)], 
                                                                test_data['FINAL GRADE'])]






for i in range(0, len(model)):
    #print 'Model ' + str(i+1)
    students['Model '+str(i+1)] = model[i].predict(students)
    #print round(students['Model '+str(i+1)][27], 2)
    

# Method For Calculating the predicted marks across various models
# given the student ID
def calculate_predicted_marks(StudentID):

    ID = StudentID-1
    marks2 = []
    
    for i in range(0, len(model)):
        marks2 = marks2 + [round(students["Model "+ str(i+1)][ID], 2)]
    
    return marks2

# The differences between the original grade and the predicted grade
# across all the models are calculated
# and a list is returned

def calculate_differences(StudentID):
    ID = StudentID
    diff2 = []
    marks2 = calculate_predicted_marks(ID)
    original = students['FINAL GRADE'][ID-1]
    
    for i in range(0, len(model)):
        diff2 = diff2 + [abs(round(original - marks2[i], 2))]
    return diff2

Matrix = [[0 for x in range(0, len(total_features))] for x in range(0, len(students['FINAL GRADE'])+1)] 

def update_lists(StudentID):
    Matrix[StudentID][0] = calculate_differences(StudentID)
    
for i in range(0, len(students)):
    update_lists(i)

Matrix[60][0] = Matrix[0][0]
# Finding the minimum difference and the index of that occurance
# using 3-d list to do the above mentioned procedure for all the
# students in a single go

#Initializing lists
minimum_value = []
minimum_index = []

for i in range(1, len(students)+1):
    minimum = Matrix[i][0][0]
    index = 1
    for j in range(0, len(model)):
        if(minimum > Matrix[i][0][j]):
            index = j
            minimum = Matrix[i][0][j]
    
    minimum_value = minimum_value + [Matrix[i][0][index]]
    minimum_index = minimum_index + [index]


import numpy as np

optimal_class_mean = round(np.mean(minimum_index), 2)
optimal_class_median = np.median(minimum_index)

def marks_predicted_mean_median(StudentID):
    i = StudentID-1
    model_number = minimum_index[i]
    origi = int(students['FINAL GRADE'][i])
    pred = round(students['Model '+str(model_number+1)][i], 2)
    diff = round(abs(origi - pred), 2)
    mean_indexed_score = round(students['Model '+str(int(optimal_class_mean))][i], 2)
    median_indexed_score = round(students['Model '+str(int(optimal_class_median))][i], 2)
    
    return origi, pred, diff, mean_indexed_score, median_indexed_score
                                               

def final_grade_list_create(StudentID):
    final_grade_list = []
    for i in range(0, len(total_features)):
        final_grade_list = final_grade_list + [students['FINAL GRADE'][StudentID-1]]
    
    return final_grade_list

def plot_graph(StudentID):
    plt.plot(calculate_predicted_marks(StudentID), '-', 
        final_grade_list_create(StudentID), '-')


students['model_multi_1'] = model_multi_1.predict(students)
students['model_multi_2'] = model_multi_2.predict(students)
students['model_multi_3'] = model_multi_3.predict(students)
students['model_multi_4_ridge'] = model_multi_4_ridge.predict(students)
students['regular_fit_all_quizzes'] = regular_fit_all_quizzes.predict(students)
students['ridge_fit_all_quizzes'] = ridge_fit_all_quizzes.predict(students)
students['mids_and_case_studies_fit'] = mids_and_case_studies_fit.predict(students)

students['model_multi_1'] = model_multi_1.predict(students)
students['model_multi_2'] = model_multi_2.predict(students)
students['model_multi_3'] = model_multi_3.predict(students)
students['model_multi_4_ridge'] = model_multi_4_ridge.predict(students)
students['regular_fit_all_quizzes'] = regular_fit_all_quizzes.predict(students)
students['ridge_fit_all_quizzes'] = ridge_fit_all_quizzes.predict(students)
students['mids_and_case_studies_fit'] = mids_and_case_studies_fit.predict(students)

def post_hoc_method_predictions(StudentID):
    ID = StudentID-1
    mm1_value = round(students['model_multi_1'][ID], 2)
    mm2_value = round(students['model_multi_2'][ID], 2)
    mm3_value = round(students['model_multi_3'][ID], 2)
    mm4r_value = round(students['model_multi_4_ridge'][ID], 2)
    rfaq_value = round(students['regular_fit_all_quizzes'][ID], 2)
    rifaq_value = round(students['ridge_fit_all_quizzes'][ID], 2)
    macsf_value = round(students['mids_and_case_studies_fit'][ID], 2)
    
    return mm1_value, mm2_value, mm3_value, mm4r_value, rfaq_value, rifaq_value, macsf_value

def temporal_predictions(StudentID):
    print 'Assessment' + '   ' + 'Marks'
    ID = StudentID-1
    for i in range(0, len(model)):
        students['Model '+str(i+1)] = model[i].predict(students)
        print '    ' +str(i+1) + '       ', round(students['Model '+str(i+1)][ID], 2)

def student_number_enter():
    number = input("Enter Student ID: ")
    return number

def check_ID_number(StudentID):
    if StudentID > 60 or StudentID == 0:
        print "Invalid ID. Please Enter Student ID between 1 - 60"
        student_number_enter()
        return 1
    else:
        return 0


def implement_all_methods(StudentID):
    if(check_ID_number(StudentID) != 1):
        ID = StudentID
    
        Original = students['FINAL GRADE'][ID-1]
        print "Final Grade Achieved (Original) :" +str(Original) 
        print "\nTemporal Predictions\n"
        temporal_predictions(ID)
    
        print "\nInterventions (if any)\n"
        intervention_prediction_on_quizzes(ID)
    
        marks_predicted_mean_median_list = marks_predicted_mean_median(ID)
        print "\nModels Derived From Temporal Predictions\n"
        print "Best Estimated Value: "+str(marks_predicted_mean_median_list[1])
        print "Best Estimated Value Was Addressed After Assessment Number: " +str(minimum_index[ID-1]+1)
        print "Difference Between Original and Best Predicted Value: " +str(marks_predicted_mean_median_list[2])
        print "Estimated Final Marks Based on Mean Assessment Indexes: " +str(marks_predicted_mean_median_list[3]) 
        print "Estimated Final Marks Based on Median Assessment Indexes: " +str(marks_predicted_mean_median_list[4])
    
    
        print "\n"
        post_hoc_method_predictions_list = post_hoc_method_predictions(ID)
        print "Post Hoc Predictions \n"
        print "Quiz Average and Mid Average: " + str(post_hoc_method_predictions_list[0])
        print "High Error Quiz Average, Mid Average, and Quiz With Highest Error Rate: " + str(post_hoc_method_predictions_list[1])
        print "Case Study Average, Mid Average, Assignment Average: " + str(post_hoc_method_predictions_list[2])
        print "Case Study Average, Mid Average, Assignment Average (Adjusted): " +str(post_hoc_method_predictions_list[3])
        print "All Quizzes: " +str(post_hoc_method_predictions_list[4])
        print "All Quizzes (Adjusted): " +str(post_hoc_method_predictions_list[5])
        print "Mid Exams and Case Studies: " +str(post_hoc_method_predictions_list[6])
    
        print "\nGraphical Representation of Data\n"
        print "Blue Line Shows The Variability Across Temporal Predictions"
        print "Green Line Shows the Final (Original) Grade"
        plot_graph(ID)
    
def marks_predicted_mean_median_display(StudentID):
        ID = StudentID
        Original = students['FINAL GRADE'][ID-1]
        marks_predicted_mean_median_list = marks_predicted_mean_median(ID)
        print "\nModels Derived From Temporal Predictions\n"
        print "Original Value: " +str(Original)
        print "Best Estimated Value: "+str(marks_predicted_mean_median_list[1])
        print "Best Estimated Value Was Addressed After Assessment Number: " +str(minimum_index[ID-1]+1)
        print "Difference Between Original and Best Predicted Value: " +str(marks_predicted_mean_median_list[2])
        print "Estimated Final Marks Based on Mean Assessment Indexes: " +str(marks_predicted_mean_median_list[3]) 
        print "Estimated Final Marks Based on Median Assessment Indexes: " +str(marks_predicted_mean_median_list[4])
    
def post_hoc_method_predictions_display(StudentID):
    ID = StudentID
    Original = students['FINAL GRADE'][ID-1]
    post_hoc_method_predictions_list = post_hoc_method_predictions(ID)
    print "Original Value: " +str(Original)
    print "Quiz Average and Mid Average: " + str(post_hoc_method_predictions_list[0])
    print "High Error Quiz Average, Mid Average, and Quiz With Highest Error Rate: " + str(post_hoc_method_predictions_list[1])
    print "Case Study Average, Mid Average, Assignment Average: " + str(post_hoc_method_predictions_list[2])
    print "Case Study Average, Mid Average, Assignment Average (Adjusted): " +str(post_hoc_method_predictions_list[3])
    print "All Quizzes: " +str(post_hoc_method_predictions_list[4])
    print "All Quizzes (Adjusted): " +str(post_hoc_method_predictions_list[5])
    print "Mid Exams and Case Studies: " +str(post_hoc_method_predictions_list[6])

def choose_module():
    print "Choose a module: "
    print "1. Interventions"
    print "2. Temporal Predictions"
    print "3. Models Derived From Temporal Predictions"
    print "4. Post Hoc Predictions"
    print "5. Graphical Representation"
    print "6. Total Data Analysis Report\n"
    
    choice = input("Enter Your Choice: ")
    if choice < 7:
        StudentID = student_number_enter()
        if StudentID < 61 and StudentID > 0:
            implement_module(choice, StudentID)
        else:
            print "Wrong ID. Please Enter A Number Between 1 - 60"
            StudentID = student_number_enter()
            implement_module(choice, StudentID)
    else:
        print "Wrong Option"
        choose_module()
    
def implement_module(choice, StudentID):
    ID = StudentID
    if choice == 1:
        print "Interventions (If any) \n"
        intervention_prediction_on_quizzes(ID)
    elif choice == 2:
        print "\nTemporal Predictions: \n"
        temporal_predictions(ID)
    elif choice == 3:
        marks_predicted_mean_median_display(ID)
    elif choice == 4:
        print "\nPost Hoc Predictions: \n"
        post_hoc_method_predictions_display(ID)
    elif choice == 5:
        print "Graphical Representation of Data: \n"
        print "Blue Line Shows The Variability Across Temporal Predictions"
        print "Green Line Shows the Final (Original) Grade"
        plot_graph(ID)
    elif choice == 6:
        print "\nTotal Data Analysis Report"
        implement_all_methods(ID)
    else:
        choose_module()

