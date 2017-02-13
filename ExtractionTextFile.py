### This file is to extract all the text in separately into a document. Basing upon that we are going to perform the word2vec conversion.
import glob
import pandas as pd
import json
import csv

path = r'annotations'  # use your path
allFiles = glob.glob(path + "/*.csv")

for file in allFiles:

    dictCollection = {"Action First-Party":[],
                      "Purpose":[],
                      "Personal Information Type":[]
                      }
    purposes = []
    personal= []
    action_first = []

    df = pd.read_csv(file, thousands=',', header=None)
    len(df)
    # df_tail = df.tail(1)[4]
    # print(df_tail)
    number_of_segments = len(df) + 1

    file = file.split('annotations/', 1)[1]

    myFile = open('collections/' + file, 'a', newline='')
    wr = csv.writer(myFile)
    # print(file)
    # print(number_of_segments)
    for i in range(number_of_segments-1):
        # print(i)
        if df[5][i] == "First Party Collection/Use":
            parse_json = json.loads(str(df[6][i]))
            if parse_json["Action First-Party"]["endIndexInSegment"] != -1:
                action_first.append(parse_json["Action First-Party"]["selectedText"])
            if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
                personal.append(parse_json["Personal Information Type"]["selectedText"])
            if parse_json["Purpose"]["endIndexInSegment"] != -1:
                purposes.append(parse_json["Purpose"]["selectedText"])

    print("dict collection ------------------------")
    print(action_first)
    wr.writerow(["ID","Text"])
    wr.writerow(["Purpose", list(purposes)])
    wr.writerow(["Action First-Party", action_first])
    wr.writerow(["Personal Information Type", personal])


