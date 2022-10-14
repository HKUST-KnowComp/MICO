import numpy as np
from collections import defaultdict
import json

CS_RELATIONS_2NL = {
    "HasPainIntensity": "causes pain intensity of",
    "HasPainCharacter": "causes pain charater of",
    "IsA": "is a",
    "NotIsA": "is not a",
    "AtLocation": "located or found at or in or on",
    "CapableOf": "is or are capable of",
    "NotCapableOf": "is not or are not capable of",
    "Causes" : "causes",
    "CausesDesire": "makes someone want",
    "CreatedBy": " is created by",
    "Desires": "desires",
    "DesireOf": "desires",
    "DefinedAs": "is defined as",
    "RelatedTo": "is related to",
    "InheritsFrom": "inherits from",
    "SymbolOf": "is a symbol of",
    "NotMadeOf": "is not made of",
    "LocatedNear": "is located near",
    "LocationOfAction": "is acted at the location of",
    "HasA": "has, possesses, or contains",
    "NotHasA": "does not have, possess, or contain",
    "HasFirstSubevent": "begins with the event or action",
    "HasLastSubevent": "ends with the event or action",
    "HasPrerequisite": "to do this, one requires",
    "HasProperty": "can be characterized by being or having",
    "NotHasProperty": "can not be characterized by being or having",
    "HasSubevent" : "includes the event or action",
    "HinderedBy" : "can be hindered by",
    "InstanceOf" : " is an example or instance of",
    "isAfter" : "happens after",
    "isBefore" : "happens before",
    "isFilledBy" : "blank can be filled by",
    "MadeOf": "is made of",
    "MadeUpOf": "made up of",
    "MotivatedByGoal": "is a step towards accomplishing the goal",
    "NotDesires": "do not desire",
    "ObjectUse": "used for",
    "UsedFor": "used for",
    "oEffect" : "as a result, PersonY or others will",
    "oReact" : "as a result, PersonY or others feel",
    "oWant" : "as a result, PersonY or others want",
    "PartOf" : "is a part of",
    "ReceivesAction" : "can receive or be affected by the action",
    "xAttr" : "PersonX is seen as",
    "xEffect" : "as a result, PersonX will",
    "xReact" : "as a result, PersonX feels",
    "xWant" : "as a result, PersonX wants",
    "xNeed" : "but before, PersonX needed",
    "xIntent" : "because PersonX wanted",
    "xReason" : "because",
    "general Effect" : "as a result, other people or things will",
    "general Want" : "as a result, other people or things want to",
    "general React" : "as a result, other people or things feel",
    # inversed
    "HasPainIntensity inversed": "is the pain intensity caused by",
    "HasPainCharacter inversed": "is the pain character caused by",
    "IsA inversed": "includes",
    "NotIsA inversed": "does not include",
    "AtLocation inversed": "is the position of",
    "CapableOf inversed": "is a skill of", # "is or are capable of"
    "NotCapableOf inversed": "is not a skill of", # "is or are capable of"
    "Causes inversed" : "because", # causes
    "CausesDesire inversed": "because", # "makes someone want",
    "CreatedBy inversed": "create", # "is created by",
    "Desires inversed": "is desired by", # "desires",
    "DesireOf inversed": "is desired by",
    "DefinedAs inversed": "is known as",
    "RelatedTo inversed": "is related to",
    "InheritsFrom inversed": "hands down to",
    "SymbolOf inversed": "can be represented by",
    "NotMadeOf inversed": "is not used to make",
    "LocatedNear inversed": "is located near",
    "LocationOfAction inversed": "is the location for acting",
    "HasA inversed": "is possessed by",# "has, possesses, or contains",
    "NotHasA inversed": "is not possessed by",# "has, possesses, or contains",
    "HasFirstSubevent inversed": "is the beginning of", # "begins with the event or action",
    "HasLastSubevent inversed": "is the end of", # "ends with the event or action",
    "HasPrerequisite inversed": "is the prerequisite of",# "to do this, one requires",
    "HasProperty inversed": "is the property of", # "can be characterized by being or having",
    "NotHasProperty inversed": "is not the property of", # "can be characterized by being or having",
    "HasSubevent inversed" : "is included by",# "includes the event or action",
    "HinderedBy inversed" : "hinder", #"can be hindered by",
    "InstanceOf inversed" : "include", #" is an example or instance of", not sure about this.
    "isAfter inversed" : "happens before", # "happens after",
    "isBefore inversed" : "happens after", # "happens before",
    "isFilledBy inversed" : "can fill",# "blank can be filled by",
    "MadeOf inversed": "make up of", # "is made of", 
    "MadeUpOf inversed": "is made of", # "made up of",
    "MotivatedByGoal inversed": "motivate", # "is a step towards accomplishing the goal",
    "NotDesires inversed": "is not desired by", # "do not desire",
    "ObjectUse inversed": "could make use of", # "used for",
    "UsedFor inversed": "could make use of", # "used for",
    "oEffect inversed" : ["PersonY or others will", "because PersonX"], #"as a result, PersonY or others will",
    "oReact inversed" : ["PersonY or others feel", "because PersonX"], #"as a result, PersonY or others feel",
    "oWant inversed" : ["PersonY or others want", "because PersonX"], # "as a result, PersonY or others want to",
    "PartOf inversed" : "include", # "is a part of",
    "ReceivesAction inversed" : "affect", # "can receive or be affected by the action",
    "xAttr inversed" : ["PersonX is seen as", "because PersonX"], # "PersonX is seen as",
    "xEffect inversed" : ["PersonX will", "because PersonX"], # "as a result, PersonX will",
    "xReact inversed" : ["PersonX feels", "because PersonX"], # "as a result, PersonX feels",
    "xWant inversed" : ["PersonX wants", "because PersonX"],# "as a result, PersonX wants to",
    "xNeed inversed" : ["PersonX needs", "as a result PersonX"],# "but before, PersonX needed",
    "xIntent inversed" : ["PersonX wanted", "as a result PersonX"], # "because PersonX wanted",
    "xReason inversed" : "as a result,",# "because",
    "general Effect inversed" : "because", # "as a result, other people or things will",
    "general Want inversed" : "because", # "as a result, other people or things want to",
    "general React inversed" : "because", # "as a result, other people or things feel",
}


def transform_two_side(inputfile, outputfile):

    dicts = defaultdict(int)

    # preprocess training dataset
    ALL = []
    ind = 0
    for line in open(inputfile):
        info = line.strip().split('\t')
        if len(info) == 2:
            continue
        head, rel, tail = info
        rel_text = CS_RELATIONS_2NL[rel].replace('PersonX', 'John').replace('PersonY', 'Tom')

        new_head = head + '. ' + rel_text
        dicts[new_head] += 1
        ALL.append((new_head, tail, ind, 'right'))
        ind += 1

        #reverse
        rel_reverse = rel + ' inversed'
        rel_reverse_text = CS_RELATIONS_2NL[rel_reverse][0].replace('PersonX', 'John').replace('PersonY', 'Tom')
        rel_reverse_text2 = CS_RELATIONS_2NL[rel_reverse][1].replace('PersonX', 'John').replace('PersonY', 'Tom')
        if rel in ['xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent', 'xNeed', 'oEffect', 'oReact', 'oWant']:
            new_tail = rel_reverse_text + ' ' + tail + '. ' + rel_reverse_text2

        dicts[new_tail] += 1
        new_head = ' '.join(head.split(' ')[1:])
        ALL.append((new_tail, new_head, ind, 'left'))
        ind += 1

    print('inputfile length: ', ind)

    # tag dict keys
    dicts_tags = {}
    tag = 0
    for key in dicts.keys():
        dicts_tags[key] = tag
        tag += 1

    print('different heads: ', tag)
    fw = open(outputfile, 'w')
    for info in ALL:
        fw.write('{}\t{}\t{}\t{}\t{}\n'.format(info[0], info[1], dicts_tags[info[0]], info[2], info[3]))
    fw.close()
 
    return dicts, ALL, ind

        
def transform_two_side_CN(inputfile, outputfile):

    dicts = defaultdict(int)
    ALL = []
    ind = 0
    for line in open(inputfile):
        info = line.strip().split('\t')
        if len(info) == 2:
            continue
        rel, head, tail = info
        rel_text = CS_RELATIONS_2NL[rel]

        new_head = head + ' ' + rel_text
        dicts[new_head] += 1
        ALL.append((new_head, tail, ind, 'right'))
        ind += 1

        #reverse
        rel_reverse = rel + ' inversed'
        rel_reverse_text = CS_RELATIONS_2NL[rel_reverse]
        new_tail = tail + ' ' + rel_reverse_text
        dicts[new_tail] += 1
        ALL.append((new_tail, head, ind, 'left'))
        ind += 1

    print('inputfile length: ', ind)

    # tag dict keys
    dicts_tags = {}
    tag = 0
    for key in dicts.keys():
        dicts_tags[key] = tag
        tag += 1

    print('different heads:', tag)
    fw = open(outputfile, 'w')
    for info in ALL:
        fw.write('{}\t{}\t{}\t{}\t{}\n'.format(info[0], info[1], dicts_tags[info[0]], info[2], info[3]))
    fw.close()
 
    return dicts, ALL, ind


if __name__ == "__main__":

    dicts, _, _ = transform_two_side('./dataset_only/ATOMIC-Ind/train.txt', 'ATOMIC-Ind-train.txt')
    dicts, _, _ = transform_two_side('./dataset_only/ATOMIC-Ind/valid.txt', 'ATOMIC-Ind-valid.txt')
    dicts, _, _ = transform_two_side('./dataset_only/ATOMIC-Ind/test.txt', 'ATOMIC-Ind-test.txt')

    dicts, _, _ = transform_two_side_CN('./dataset_only/CN-82K-Ind/train.txt', 'CN82k-Ind-train.txt')
    dicts, _, _ = transform_two_side_CN('./dataset_only/CN-82K-Ind/valid.txt', 'CN82k-Ind-valid.txt')
    dicts, _, _ = transform_two_side_CN('./dataset_only/CN-82K-Ind/test.txt', 'CN82k-Ind-test.txt')

