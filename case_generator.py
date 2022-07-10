import itertools
import random
from random import randint
from typing import Any, List, Tuple, Dict, Union, Callable
import numpy as np

class Param_Wrapper():

    def __init__(self, value: Any, restrictions: List[Tuple[int, int, int]] = []) -> None:
        self.value = value
        self.restrictions = restrictions #list_idx, set_idx, restriction_type(1 included, 0 excluded)

    @staticmethod
    def convert_1d_idx_to_2d(d1_val: str, D2_list: List[list]) -> Tuple[int, int]:
        """takes 1D value and finds the 2D equivalent and returns that element in passed D2_list."""
        int_d1 = int(d1_val)
        idx1 = 0
        sum_idx = 0
        dims = (len(inner_list) for inner_list in D2_list)

        for dim_size in dims:
            if sum_idx + dim_size > int_d1:
                break

            sum_idx += dim_size
            idx1+=1 
        
        idx2 = int_d1 - sum_idx

        return idx1, idx2
    
    @staticmethod
    def set_dim_converter(d1_val: str, D2_list: List[list]):
        idx1, idx2 = Param_Wrapper.convert_1d_idx_to_2d(d1_val, D2_list)
        return D2_list[idx1][idx2]

    def legal_set(self, p_set: set, o_set_dims: tuple) -> bool:
        """Checks if this wrapper can legally belong to passed set.
        
        Args:
            p_set: set this object belongs to
            o_set_dims:  dimensions of original 2D list
        
        returns: bool indicating if legal"""
        if len(self.restrictions) == 0: #no restriction always legal
            return True
        include_set: set = set() #obj that must be in same set
        exclude_set: set = set() #obj that cannot be in same set

        for restriction in self.restrictions:
            #converts from 2D representation used in restriction to 1D used in sets
            set_val = sum([val for idx, val in enumerate(o_set_dims) if idx < restriction[0]]) + restriction[1]
            str_set_val = str(set_val) #needs to be a str before being added
            
            if restriction[2] == 1:
                include_set.add(str_set_val)
            elif restriction[2] == 0:
                exclude_set.add(str_set_val)
            else:
                raise ValueError("passed bad restriction type value {0}. restriction types can only be 0: exclusive or 1: inclusive".format(restriction[2]))
        
        valid_exclude_set = len(exclude_set & p_set) == 0 #make sure we have no elements from exclude set
        return include_set.issubset(p_set) and valid_exclude_set

def wrap_obj(obj: Any):
    """Wrapped passed object so it can be used in set generation"""

    #Check for valid restriction format
    if type(obj) is list or type(obj) is tuple: #indexable by Param_Wrapper
        if len(obj) == 2: 
            if type(obj[1]) is list: #1st index is list
                valid_tuple = True

                for tupl in obj[1]:
                    valid_tuple = valid_tuple and  tuple is type(tupl) #restriction must be passed as tuple
                    valid_tuple = valid_tuple and (len(tupl) == 3) #valid restriction tuple is size 3
                
                if valid_tuple:
                    return Param_Wrapper(obj[0], obj[1])

    #otherwise just wrap with no restriction
    return Param_Wrapper(obj)
    
def wraps_param_vars(unwrapped_vars: List[List]) -> List[List[Param_Wrapper]]:
    """wraps nested list of variables so they can be used in our set comparison operations"""
    wrapped_nested_list = [None] * len(unwrapped_vars)
    for i, param_list in enumerate(unwrapped_vars):

        wrapped_list = [None] * len(param_list)
        for j, param in enumerate(param_list):
            wrapped_list[j] = wrap_obj(param)
        
        wrapped_nested_list[i] = wrapped_list
    
    return wrapped_nested_list

def prune_sets(param_vars: List[List[Param_Wrapper]], sets: List[set]):
    """Modifies passed sets by removing any set that contains a param whose restrictions are violated"""
    dims = tuple([len(inner_list) for inner_list in param_vars])
    for idx, n_set in enumerate(sets):
        
        for element in n_set:
            
            val: Param_Wrapper = Param_Wrapper.set_dim_converter(element, param_vars) #get param that maps to this set value
            legal_set = val.legal_set(n_set, dims) #check if this param is legally allowed in this set

            if not legal_set:
                sets[idx] = None #if any element is invalid the entire set is illegal
                break            #only need one illegal case so we are done for this set
 
    sets = [value for value in sets if value != None] #remove empty idxs
    return sets

def set_generator(param_vars: List[List[Param_Wrapper]]):
    counter = 0
    var_sets = [None] * len(param_vars)
    dims = () #tracks dimensions of inner lists

    for i, var_list in enumerate(param_vars):
        sub_var_set = [None] * len(var_list)

        for j, _ in enumerate(var_list):
            sub_var_set[j] = str(counter)
            counter += 1

        var_sets[i] = sub_var_set
        dims += (len(var_list),) # add length of inner list to dims
       
    var_sets = [set(val)  for val in itertools.product(*var_sets)] #get iterable product of all our combinations
    return var_sets


a= ["hi", "bye"]
b = ["no", "way", "jose"]
c = ["good", "time"]
d = [a,b,c]
e = wraps_param_vars(d)
test = set_generator(e)
pruned =  prune_sets(e, test)
print(test)

def new_get_vars(vars: Union[Any, Tuple[Callable, dict]]) -> list:
    """Allows passing of fxns w/args in unit tests.

    If passing a fxn must be in a tuple with following format:
    Tuple(fxn_name, dict{arg1_name: arg1_val, ..., argN_name: argN_val}"""

    called_vars = []
    for _, var in enumerate(vars):
        if type(var) is tuple:
            assert callable(var[0]), "first element of tuple passed to arg generator must be a fxn"

            if len(var[1]) == 0:            
                called_vars.append(var[0]()) #no args so we can just eval
            else:
                called_vars.append(var[0](**var[1])) #pass arguments in dict to our fxn
        else:
            called_vars.append(var)
    return called_vars

def combination_w_fixed_elements(vals: List[List[Any]], fixed_elm: List[Any]):
    """creates list of tuples representing all possible combinations between non_fixed elements in vals"""
    #make sure at least one fixed elm is in every idx of vals
    for elm in fixed_elm:
        for val in vals:
            assert fixed_elm in val, "asked to fix element {0}, but did not find it in list {1}".format(elm, vals)

    combo_vals = list(itertools.product(*vals)) #get all val list combinations

    #remove any combinations that does not contain EVERY fixed element.
    for combo in combo_vals:
        
        for elm in fixed_elm:
            if elm not in combo:
                combo_vals.remove(combo)
                break #we don't need to keep looking once we found 1 violation in current combo

    return combo_vals
        

def _get_vars(vars: List[Any]) -> list:
    """Helper method for tuple_truth_table_generator"""
    #If any variable in vars is a function we need to evaluate it first.
    called_vars = []
    for idx, var in enumerate(vars):
        if callable(var):
            called_vars.append(var())
        else:
            called_vars.append(var)
    return called_vars

def new_tuple_truth_table_generator(num_bool: int, fixed_elm: List[Tuple[int, bool]]) -> list:
    num_tuples = pow(2, num_bool)
    tuple_lists = [None for _ in range(num_tuples)]

    for i in range(num_tuples):
        valid_add = True

        # convert decimal to binary list
        res = np.array([int(j) for j in list('{0:0b}'.format(i))])
        pad_len = num_bool - len(res)
        pad_res = np.pad(res, (pad_len, 0), 'constant', constant_values=0) # left pad list with zeros to desired length
        pad_res = pad_res.astype(bool)  # Convert to boolean list

        #make sure we don't have illegal vals as defined by fixed_elm
        for f_elm in fixed_elm:
            if pad_res[f_elm[0]] != f_elm[1]:
                valid_add = False
                break

        if valid_add:

            tuple_lists[i] = tuple(pad_res)
    
    return tuple_lists

def tuple_truth_table_generator(vars: List[Any], num_bool: int) -> list:
    """
    So the goal of this function is to basically help with making parameterization less clunky.
    It basically creates a True/False table for vars with num_bools elements.
    Args:
        vars: arbitrary list of anything
        bool: number of booleans desired
    """
    num_tuples = pow(2, num_bool)
    tuple_lists = [None for _ in range(num_tuples)]

    for i in range(num_tuples):
        # convert decimal to binary list
        res = np.array([int(j) for j in list('{0:0b}'.format(i))])
        pad_len = num_bool - len(res)
        # left pad list with zeros to desired length
        pad_res = np.pad(res, (pad_len, 0), 'constant', constant_values=0)
        pad_res = pad_res.astype(bool)  # Convert to boolean list
        new_vars = _get_vars(vars) + list(pad_res)
        tuple_lists[i] = tuple(new_vars)
    
    return tuple_lists

