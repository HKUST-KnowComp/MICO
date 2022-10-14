import xml.etree.ElementTree as etree
import json
import numpy as np
import os

def transform(split):
    tree = etree.parse('./COPA-resources/datasets/copa-{}.xml'.format(split))
    root = tree.getroot()
    original_problems = root.getchildren()

    fw = open('copa-{}-new.jsonl'.format(split), 'w')

    for original_problem in original_problems:
        problem = dict()
        first_info = original_problem.attrib
        problem["qID"] = first_info['id']
        problem["answer"] = first_info['most-plausible-alternative']
        problem_type = first_info['asks-for']

        info = original_problem.getchildren()
        if info[0].tag == 'p':
            if problem_type == 'cause':
                problem["sentence"] = info[0].text + ' The cause for it was that _'
            elif problem_type == 'effect':
                problem["sentence"] = info[0].text + ' As a result, _'
        if info[1].tag == 'a1':
            problem["option1"] = info[1].text
        if info[2].tag == 'a2':
            problem["option2"] = info[2].text
        fw.write('{}\n'.format(json.dumps(problem)))

    fw.close()


if __name__ == "__main__"

    #spplit is dev or test
    split = 'dev'

    transform(split)
