#!/usr/bin/python3
description = '''
Parses the resulting parameters from csv output of parameter_seach.py,
Nicely displayed with "$./read_results.py file.csv | column -s, -t"
'''

import argparse
import sys
import csv
import ast

parser = argparse.ArgumentParser(description=description)
parser.add_argument('csv_file', type=str, help='.csv file from parameter_search.py')
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('-p', '--parameters', action='store_true', help='Show hyper-parameters used in search [deafault]')
group.add_argument('-r', '--results', action='store_true', help='Show resulting metrics and loss')
args = parser.parse_args()


f = open(args.csv_file, 'r')
reader = csv.reader(f)
next(reader)

if not args.results:
    print("Result, iter, optimizer, learning rate, SGD momentum, age loss function, sex loss weight, batch size, dropout rates")
    for line in reader:
        iteration = line[3]
        sex_acc = line[0]
        params = ast.literal_eval(line[1])
        batch_size = int(params['batch_size'])

        dropout_rate=str([params['dropout_rate1'],
                      params['dropout_rate2'],
                      params['dropout_rate3'],
                      params['dropout_rate4'],
                      params['dropout_rate5']]).replace(",", " ")


        learning_rate = params['optimizer']['learning_rate']
        optimizer = params['optimizer']['optimizer']
        if optimizer == 'SGD':
            momentum = params['optimizer']['momentum']
        else:
            momentum = 'null'

        age_loss = params['age_loss']
        sex_loss_weight = params['sex_loss_weight']

        print(sex_acc, iteration, optimizer, learning_rate, momentum, age_loss, sex_loss_weight, batch_size, dropout_rate, sep=', ')

if args.results:
	print("sex validation accuracy, sex training accuracy, iter,",
	      "age val MAE, age train MAE, sex val loss,",
	      "sex train loss, val loss, train loss")
	for line in reader:
	    iteration = line[3]
	    sex_acc = line[0]
	    history = line[2].replace("nan", "'nan'").replace("inf", "'inf'")
	    h = ast.literal_eval(history)
	    print(h['val_sex_masked_accuracy'][-1], h['sex_masked_accuracy'][-1], iteration,
		  h['val_age_masked_mae'][-1], h['age_masked_mae'][-1], h['val_sex_loss'][-1],
		  h['sex_loss'][-1], h['val_loss'][-1], h['loss'][-1], sep=', ')
f.close()
