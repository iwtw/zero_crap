import os
import datetime


print('MAIN:CLEANING DATA')
os.system('./preprocess_data/clean_all.sh')
print('MAIN:TRAINING')
time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_info = '../data/bagging_results/' + time + '.txt'
#debugging
#os.system('python bagging.py -num_models 1 -output_list ' + model_info  )
#final
os.system('python bagging.py -output_list ' + model_info  )
print('MAIN:TESTING')
os.system('mkdir -p ../submit')
os.system('python test_ensemble_gcn.py {} --output_list {} '.format(  model_info , '../submit/submit_' + time + '.txt'  ) )

