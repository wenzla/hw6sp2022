import math
import sys

train_y = []
train_x = {}
test_x = {}

with open('dt_debug_testcases/input22.txt') as f:
    lines = [line.rstrip() for line in f]
    
with open('dt_debug_testcases/output22.txt') as f:
    answer = [line.rstrip() for line in f]

# for line in sys.stdin:
for line in lines:
    sep_line = (line.split(' '))
    if int(sep_line[0]) != -1: #-1 is not training label
        train_y.append(sep_line[0])
        for attr in sep_line[1:]:
            name, val = attr.split(':')
            if name not in train_x.keys():
                train_x[name] = [float(val)]
            else:
                train_x[name].append(float(val))
    else:
        for attr in sep_line[1:]:
            name, val = attr.split(':')
            if name not in test_x.keys():
                test_x[name] = [float(val)]
            else:
                test_x[name].append(float(val))

class DTree(): #Tree class
    def __init__(self,attr,threshold,is_leaf=False, left=None):
        self.left = None
        self.right = None
        self.is_leaf = is_leaf
        self.name = attr
        self.theta = threshold
        
        if self.is_leaf:
            self.left=left
        
    def predict(self, row):
        if not self.is_leaf:
            if row[self.name] <= self.theta:
                return self.left.predict(row)
            else:
                return self.right.predict(row)
        else:
            return self.left

def get_common(lst):
    pos_lst_vals = list(set(lst))
    
    common_val = -1
    common_label = '9999999999'
    for plv in pos_lst_vals:
        if lst.count(plv) > common_val:
            common_val = lst.count(plv)
            common_label = plv
        elif lst.count(plv) == common_val: 
            if int(plv) < int(common_label):
                common_label = plv
    return common_label

def keymaxval(d):
    v = list(d.values())
    k = list(d.keys())
    
    max_label = "999999"
    max_val = -1
    for key in k:
        if d[key] > max_val:
            max_label = key
            max_val = d[key]
        elif d[key] == max_val:
            if int(key) < int(max_label):
                max_label = key
                max_val = d[key]
    return max_label

def get_attr_thresh(x,y,find_ideal=False):
    threshs = {}
    for key in x.keys():
        pos_vals = list(set([round(v,1) for v in x[key]]))
        pos_vals.sort()
        possible_threshes = []
        if len(pos_vals) == 1:
            possible_threshes.append(pos_vals[0])
        for i in range(len(pos_vals)-1):
            possible_threshes.append((pos_vals[i]+pos_vals[i+1])/2)
        max_info = 0
        key_thresh = 0
#         first = True
        for pt in possible_threshes:
#             if first:
#                 key_thresh = pt
#                 first = False
            temp = {key: pt}
            test_info = get_info_gains(x, y, temp)[key]
            if test_info > max_info:
                max_info = test_info
                key_thresh = pt
            elif test_info == max_info:
                if key_thresh > pt:
                    max_info = test_info
                    key_thresh = pt
        threshs[key] = key_thresh

    return threshs

def get_overall_info(actuals):
    overall_info = 0
    class_lens = {}

    unique_vals = set(actuals)
    for y in actuals:
        class_lens[y] = actuals.count(y)
        
    overall_info = 0
    for cl in class_lens.keys():
        cls_len = class_lens[cl]
        frac = (cls_len/len(actuals))
        if frac > 0:
            overall_info -= frac*math.log((frac),2)

    return overall_info, class_lens

def get_info_gains(data, actuals, threshs):
    
    overall_info, class_lens = get_overall_info(actuals)
    ipns = {}
    for key in threshs.keys():
        temp_thresh = [0 if val <= threshs[key] else 1 for val in data[key]]
        not_met_inds = [i for i, x in enumerate(temp_thresh) if x == 0]
        met_inds = [i for i, x in enumerate(temp_thresh) if x == 1]

        class_lbls_met = [actuals[i] for i in met_inds]
        class_lbls_not = [actuals[i] for i in not_met_inds]

        ipn_met = 0
        for cl in class_lens.keys():
            cls_len = class_lens[cl]
            if len(class_lbls_met) == 0:
                frac = 0
            else:
                frac = (class_lbls_met.count(cl)/len(class_lbls_met))
            if frac > 0:
                ipn_met -= frac*math.log((frac),2)
        
        ipn_not = 0
        for cl in class_lens.keys():
            cls_len = class_lens[cl]
            if len(class_lbls_not) == 0:
                frac = 0
            else:
                frac = (class_lbls_not.count(cl)/len(class_lbls_not))
            if frac > 0:
                ipn_not -= frac*math.log((frac),2)

        attr_info_gain = ((len(class_lbls_met)/len(actuals))*ipn_met) + ((len(class_lbls_not)/len(actuals))*ipn_not)
        info_gain = overall_info - attr_info_gain
        ipns[key] = info_gain
        
    return ipns

def make_left_right_split(data, actuals, threshs, biggest_gain):
    temp_thresh = [0 if val <= threshs[biggest_gain] else 1 for val in data[biggest_gain]]
    left = [i for i, x in enumerate(temp_thresh) if x == 0]
    right = [i for i, x in enumerate(temp_thresh) if x == 1]
    
    left_x = {}
    right_x = {}
    left_y = {}
    right_y = {}
    for key in data.keys():
        left_x[key] = [data[key][i] for i in left]
        right_x[key] = [data[key][i] for i in right]

    left_y = [actuals[i] for i in left]
    right_y = [actuals[i] for i in right]
    
    return left_x, right_x, left_y, right_y

def get_leaf_vals(data, actuals, threshs, biggest_gain):
    _, _, left_y, right_y = make_left_right_split(data, actuals, threshs, biggest_gain) 
    return (get_common(left_y), get_common(right_y))

threshs = get_attr_thresh(train_x,train_y, find_ideal=True)
ipns = get_info_gains(train_x, train_y, threshs)
biggest_gain = keymaxval(ipns)

model = DTree(biggest_gain,threshs[biggest_gain])

left_x, right_x, left_y, right_y = make_left_right_split(train_x, train_y, threshs, biggest_gain)

# Branches
if len(list(set(left_y))) == 1: # Stop early if needed
    model.left = DTree(1,1,True, left_y[0])
if len(list(set(right_y))) == 1:
    model.right = DTree(1,1,True, right_y[0])

if model.left is None:
    left_threshs = get_attr_thresh(left_x, left_y, find_ideal=True)
    left_ipns = get_info_gains(left_x, left_y, left_threshs)
    left_biggest_gain = keymaxval(left_ipns)
    model.left = DTree(left_biggest_gain,left_threshs[left_biggest_gain])

    left_leaf, right_leaf = get_leaf_vals(left_x, left_y, left_threshs, left_biggest_gain)
    model.left.left = DTree(1,1,True,left_leaf)
    model.left.right = DTree(1,1,True,right_leaf)
    
if model.right is None:
    right_threshs = get_attr_thresh(right_x, right_y, find_ideal=True)
    right_ipns = get_info_gains(right_x, right_y, right_threshs)
    right_biggest_gain = keymaxval(right_ipns)
    model.right = DTree(right_biggest_gain, right_threshs[right_biggest_gain])
    
    left_leaf, right_leaf = get_leaf_vals(right_x, right_y, right_threshs, right_biggest_gain)
    model.right.left = DTree(1,1,True,left_leaf)
    model.right.right = DTree(1,1,True,right_leaf)

for i in range(len(test_x[list(test_x.keys())[0]])):
    row = {}
    for key in test_x.keys():
        row[key] = test_x[key][i]
    print(model.predict(row))

