import os, shutil
import glob
import random

# path = r'annotations'  # use your path
# allFiles = glob.glob(path + "/*.csv")
#
# print(len(allFiles))


# num_validation_samples = 30
# list_of_validation_files = random.sample(allFiles, num_validation_samples)
#
# print(list_of_validation_files)

# for file in list_of_validation_files:
#     file_ = file.split('annotations/', 1)[1]
#     shutil.move(file, 'Validate/'+file_)

# now all the validation files are moved out of the folder

# num_test_samples = 20
# list_of_test_files = random.sample(allFiles, num_test_samples)
#
# print(list_of_test_files)
#
# for file in list_of_test_files:
#     file_ = file.split('annotations/', 1)[1]
#     shutil.move(file, 'Test/'+file_)


# Moving all the data into Train folder
# for file in allFiles:
#     file_ = file.split('annotations/', 1)[1]
#     shutil.move(file, 'Train/'+file_)



import glob
import pandas as pd
import json
import csv

path = r'Train'  # use your path
allFiles = glob.glob(path + "/*.csv")

# action first party
total_data_samples0 = []
total_data_number0 = []
total_data_label0 = []

# Personal Info Type
total_data_samples1 = []
total_data_number1 = []
total_data_label1 = []

# Purpose
total_data_samples2 = []
total_data_number2 = []
total_data_label2 = []

# None
total_data_samples3 = []
total_data_number3 = []
total_data_label3 = []

cnt = 0
cnt0 = 0
cnt1 = 0
cnt2 = 0
cnt3 = 0
choice = 0

print(len(allFiles))


for file in allFiles:
    print(file)
    dictCollection = {"Action First-Party": [],
                      "Purpose": [],
                      "Personal Information Type": [],
                      "None": []
                      }

    df = pd.read_csv(file, thousands=',', header=None)
    len(df)
    # df_tail = df.tail(1)[4]
    # print(df_tail)
    number_of_segments = len(df) + 1

    file = file.split('Train/', 1)[1]
    #     print(file)
    for i in range(number_of_segments - 1):
        # print(i)
        choice = 0  # whether the sentence is from one of the following categories
        if df[5][i] == "First Party Collection/Use":
            choice = 1
            parse_json = json.loads(str(df[6][i]))
            #             print(parse_json)
            if parse_json["Action First-Party"]["endIndexInSegment"] != -1:
                total_data_samples0.append(parse_json["Action First-Party"]["selectedText"])
                total_data_number0.append(0)
                total_data_label0.append("Action First-Party")
                cnt0 = cnt0 + 1
            if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
                total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
                total_data_number1.append(1)
                total_data_label1.append("Personal Information Type")
                cnt1 = cnt1 + 1
            if parse_json["Purpose"]["endIndexInSegment"] != -1:
                total_data_samples2.append(parse_json["Purpose"]["selectedText"])
                total_data_number2.append(2)
                total_data_label2.append("Purpose")
                cnt2 = cnt2 + 1
        if df[5][i] == "Third Party Sharing/Collection":
            choice = 1
            parse_json = json.loads(str(df[6][i]))
            #             print(parse_json)
            if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
                total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
                total_data_number1.append(1)
                total_data_label1.append("Personal Information Type")
                cnt1 = cnt1 + 1
            if parse_json["Purpose"]["endIndexInSegment"] != -1:
                total_data_samples2.append(parse_json["Purpose"]["selectedText"])
                total_data_number2.append(2)
                total_data_label2.append("Purpose")
                cnt2 = cnt2 + 1
        if df[5][i] == "User Choice/Control":
            choice = 1
            parse_json = json.loads(str(df[6][i]))
            #             print(parse_json)
            if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
                total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
                total_data_number1.append(1)
                total_data_label1.append("Personal Information Type")
                cnt1 = cnt1 + 1
            if parse_json["Purpose"]["endIndexInSegment"] != -1:
                total_data_samples2.append(parse_json["Purpose"]["selectedText"])
                total_data_number2.append(2)
                total_data_label2.append("Purpose")
                cnt2 = cnt2 + 1
        if df[5][i] == "Data Retention":
            choice = 1
            parse_json = json.loads(str(df[6][i]))
            #             print(parse_json)
            if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
                total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
                total_data_number1.append(1)
                total_data_label1.append("Personal Information Type")
                cnt1 = cnt1 + 1
        if choice == 0:
            #             print("practis is -->", df[5][i])
            parse_json = json.loads(str(df[6][i]))
            #             print(parse_json)
            #             print(len(df[6][i]))
            #             print(parse_json.keys())
            attributes = parse_json.keys()
            #             print(attributes)
            for k in attributes:
                #                 print(k)
                if parse_json[k]['startIndexInSegment'] != -1:
                    #                     print(parse_json[k]['selectedText'])
                    total_data_samples3.append(parse_json[k]['selectedText'])
                    total_data_number3.append(3)
                    total_data_label3.append("None")
                    cnt3 = cnt3 + 1

print("---")
print(cnt0)
print(cnt1)
print(cnt2)
print(cnt3)
print("--")
print(cnt0 + cnt1 + cnt2 + cnt3)
print(len(total_data_samples0))
print(len(total_data_samples1))
print(len(total_data_samples2))
print(len(total_data_samples3))