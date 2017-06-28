from pycorenlp import StanfordCoreNLP
from nltk.tree import Tree

nlp = StanfordCoreNLP('http://localhost:9000')

text = ('Pusheen and Smitha walked along the beach. '
        'Pusheen wanted to surf, but fell off the surfboard.')

print text

output = nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,pos,depparse,parse', 'outputFormat': 'json'})

print(output['sentences'][0]['parse'])
print(output['sentences'][1]['parse'])



# parsestr=output['sentences'][0]['parse']
# for i in Tree.fromstring(parsestr).subtrees():
#     if i.label() == 'NP':
#         print i.leaves()
