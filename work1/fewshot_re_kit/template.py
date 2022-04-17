# get template
def get_template(item):
    prompt_tokens = list()
    prompt_tokens += item['tokens']
    template = " [SEP] " + item['h'][0] + " [MASK] " + item['t'][0] + " ."
    prompt_tokens += template.split(" ")
    kg_head_pos1_index = len(item['tokens']) + 2
    kg_head_pos2_index = kg_head_pos1_index + len(item['h'][0].split(" ")) - 1
    
    kg_tail_pos1_index = kg_head_pos1_index + len(item['h'][0].split(" ")) + 1
    kg_tail_pos2_index = kg_tail_pos1_index + len(item['t'][0].split(" ")) - 1
    # print(prompt_tokens[kg_head_pos1_index], item['h'][0])
    # print(prompt_tokens[kg_tail_pos1_index], item['t'][0])
    return prompt_tokens, [kg_head_pos1_index, kg_head_pos2_index], [kg_tail_pos1_index, kg_tail_pos2_index]

def get_template1(item):
    # 句子中的entity
    prompt_tokens = list()
    prompt_tokens += item['tokens']
    template = " [SEP] " + item['h'][0] + " [MASK] " + item['t'][0] + " ."
    prompt_tokens += template.split(" ")
    kg_head_pos1_index = item['h'][2][0][0]
    kg_head_pos2_index = item['h'][2][0][-1]
    
    kg_tail_pos1_index = item['t'][2][0][0]
    kg_tail_pos2_index = item['t'][2][0][-1]
    # print(prompt_tokens[kg_head_pos1_index], item['h'][0])
    # print(prompt_tokens[kg_tail_pos1_index], item['t'][0])
    return prompt_tokens, [kg_head_pos1_index, kg_head_pos2_index], [kg_tail_pos1_index, kg_tail_pos2_index]


def get_template2(item, sep, mask):
    prompt_tokens = list()
    prompt_tokens += item['tokens']
    sentence1 = sep + " the head entity " + item['h'][0] + " is " + mask + " ."
    sentence2 = sep + " the tail entity " + item['t'][0] + " is " + mask + " ."
    sentence3 = sep + " the relation between " + item['h'][0] + " and " + item['t'][0] + " is " + mask + " ."
    
    prompt_tokens += sentence1.split(" ") + sentence2.split(" ") + sentence3.split(" ")
    mask_pos1 = len(item['tokens']) + len(item['h'][0].split(" ")) + 5
    mask_pos2 = mask_pos1 + len(item['t'][0].split(" ")) + 7
    mask_pos3 = mask_pos2 + len(item['h'][0].split(" ")) + len(item['t'][0].split(" ")) + 8
    if len(prompt_tokens) > 128:
        print(len(prompt_tokens))
    return prompt_tokens, mask_pos1, mask_pos2, mask_pos3
    
    
def get_template3(item, sep, mask):
    prompt_tokens = list()
    prompt_tokens += item['tokens']
    sentence1 = sep + " the head entity " + item['h'][0] + " is " + mask + " ."
    sentence2 = sep + " the tail entity " + item['t'][0] + " is " + mask + " ."
    sentence3 = sep + " the relation between " + item['h'][0] + " and " + item['t'][0] + " is " + mask + " ."
    
    prompt_tokens += sentence1.split(" ") + sentence2.split(" ") + sentence3.split(" ")
    mask_pos1 = len(item['tokens']) + len(item['h'][0].split(" ")) + 5
    mask_pos2 = mask_pos1 + len(item['t'][0].split(" ")) + 7
    mask_pos3 = mask_pos2 + len(item['h'][0].split(" ")) + len(item['t'][0].split(" ")) + 8
    if len(prompt_tokens) > 128:
        print(len(prompt_tokens))
    return prompt_tokens, mask_pos1, mask_pos2, mask_pos3

def get_template4(item):
    template = ["This", "sentence", "of", "\""]
    prompt_tokens = list()
    prompt_tokens = template + item['tokens'] + ["\"", "means", "[MASK]", "."]
    kg_head_pos1_index = 4 + item['h'][2][0][0]
    kg_head_pos2_index = 4 + item['h'][2][0][-1]
    
    kg_tail_pos1_index = 4 + item['t'][2][0][0]
    kg_tail_pos2_index = 4 + item['t'][2][0][-1]
    # print(prompt_tokens[kg_head_pos1_index], item['h'][0])
    # print(prompt_tokens[kg_tail_pos1_index], item['t'][0])
    return prompt_tokens, [kg_head_pos1_index, kg_head_pos2_index], [kg_tail_pos1_index, kg_tail_pos2_index]
    
def get_template5(item):
    prompt_tokens = list()
    prompt_tokens = item['tokens']
    kg_head_pos1_index = item['h'][2][0][0]
    kg_head_pos2_index = item['h'][2][0][-1]
    
    kg_tail_pos1_index = item['t'][2][0][0]
    kg_tail_pos2_index = item['t'][2][0][-1]
    # print(prompt_tokens[kg_head_pos1_index], item['h'][0])
    # print(prompt_tokens[kg_tail_pos1_index], item['t'][0])
    return prompt_tokens, [kg_head_pos1_index, kg_head_pos2_index], [kg_tail_pos1_index, kg_tail_pos2_index]