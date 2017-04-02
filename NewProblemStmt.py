import glob
import pandas as pd
import json
import csv
from bs4 import BeautifulSoup, NavigableString
import nltk.data

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
allFiles = allFiles[:1]


def html_to_text(html):
    "Creates a formatted text email message as a string from a rendered html template (page)"
    html_report_part1 = open(html, 'r')
    soup = BeautifulSoup(html_report_part1, 'html.parser')
    # Ignore anything in head
    #     print(len(list(soup.descendants)))
    text = []
    start_index = []
    end_index = []
    for element in soup.descendants:
        #         print(element)
        # We use type and not isinstance since comments, cdata, etc are subclasses that we don't want
        count = 0;
        if type(element) == NavigableString:
            # We use the assumption that other tags can't be inside a script or style
            if element.parent.name in ('script', 'style'):
                continue

            # remove any multiple and leading/trailing whitespace
            string = ' '.join(element.string.split())
            if string:
                if element.parent.name == 'a':
                    a_tag = element.parent
                    # replace link text with the link
                    string = a_tag['href']
                    # concatenate with any non-empty immediately previous string
                    if (type(a_tag.previous_sibling) == NavigableString and
                            a_tag.previous_sibling.string.strip()):
                        text[-1] = text[-1] + ' ' + string
                        continue
                elif element.previous_sibling and element.previous_sibling.name == 'a':
                    text[-1] = text[-1] + ' ' + string
                    continue
                elif element.parent.name == 'p':
                    # Add extra paragraph formatting newline
                    string = '\n' + string
                text += [string]
                #                 print(len(string))
                start_index.append(count)
                #                 print(count)
    doc = '\n'.join(text)
    return doc


for file in allFiles:
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

    ### beautifulsuop on html doc
    file_html = 'Plain_Html/' + file
    file_html = file_html.split('csv', 1)[0]
    file_html = file_html + 'html'
    print(file_html)
    ls = []
    indeces = []
    data = html_to_text(file_html)
    ls = tokenizer.tokenize(data)
    count = 0
    for l in ls:
        indeces.append([count, count + len(l)])
        count = count + len(l)
    print(len(indeces))
    print(len(ls))
    ###


    for i in range(number_of_segments - 1):
        #         print(i)
        choice = 0  # whether the sentence is from one of the following categories
        for index in range(len(l)):
            if df[5][i] == "First Party Collection/Use":
                choice = 1
                parse_json = json.loads(str(df[6][i]))
                print(parse_json)
                if parse_json["Action First-Party"]["endIndexInSegment"] != -1:
                    print("ActionFirst Party", parse_json["Action First-Party"]["startIndexInSegment"],
                          parse_json["Action First-Party"]["endIndexInSegment"])
                    total_data_samples0.append(parse_json["Action First-Party"]["selectedText"])
                    total_data_number0.append(0)
                    total_data_label0.append("Action First-Party")
                    cnt0 = cnt0 + 1
                if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
                    print("Personal Information Type", parse_json["Personal Information Type"]["startIndexInSegment"],
                          parse_json["Personal Information Type"]["endIndexInSegment"])
                    total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
                    total_data_number1.append(1)
                    total_data_label1.append("Personal Information Type")
                    cnt1 = cnt1 + 1
                if parse_json["Purpose"]["endIndexInSegment"] != -1:
                    print("Purpose", parse_json["Purpose"]["startIndexInSegment"],
                          parse_json["Purpose"]["endIndexInSegment"])
                    total_data_samples2.append(parse_json["Purpose"]["selectedText"])
                    total_data_number2.append(2)
                    total_data_label2.append("Purpose")
                    cnt2 = cnt2 + 1
# if df[5][i] == "Third Party Sharing/Collection":
#             choice = 1
#             parse_json = json.loads(str(df[6][i]))
# #             print(parse_json)
#             if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
#                 total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
#                 total_data_number1.append(1)
#                 total_data_label1.append("Personal Information Type")
#                 cnt1 = cnt1+1
#             if parse_json["Purpose"]["endIndexInSegment"] != -1:
#                 total_data_samples2.append(parse_json["Purpose"]["selectedText"])
#                 total_data_number2.append(2)
#                 total_data_label2.append("Purpose")
#                 cnt2 = cnt2+1
#         if df[5][i] == "User Choice/Control":
#             choice = 1
#             parse_json = json.loads(str(df[6][i]))
#             if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
#                 total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
#                 print(parse_json["Personal Information Type"]["startIndexInSegment"],parse_json["Personal Information Type"]["endIndexInSegment"])
#                 total_data_number1.append(1)
#                 total_data_label1.append("Personal Information Type")
#                 cnt1 = cnt1+1
#             if parse_json["Purpose"]["endIndexInSegment"] != -1:
#                 total_data_samples2.append(parse_json["Purpose"]["selectedText"])
#                 total_data_number2.append(2)
#                 total_data_label2.append("Purpose")
#                 cnt2 = cnt2+1
#         if df[5][i] == "Data Retention":
#             choice = 1
#             parse_json = json.loads(str(df[6][i]))
#             if parse_json["Personal Information Type"]["endIndexInSegment"] != -1:
#                 total_data_samples1.append(parse_json["Personal Information Type"]["selectedText"])
#                 total_data_number1.append(1)
#                 total_data_label1.append("Personal Information Type")
#                 cnt1 = cnt1+1
#         if choice==0:
# #             print("practis is -->", df[5][i])
#             parse_json = json.loads(str(df[6][i]))
# #             print(parse_json)
# #             print(len(df[6][i]))
# #             print(parse_json.keys())
#             attributes = parse_json.keys()
# #             print(attributes)
#             for k in attributes:
# #                 print(k)
#                 if parse_json[k]['startIndexInSegment'] != -1:
# #                     print(parse_json[k]['selectedText'])
#                     total_data_samples3.append(parse_json[k]['selectedText'])
#                     total_data_number3.append(3)
#                     total_data_label3.append("None")
#                     cnt3 = cnt3+1

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