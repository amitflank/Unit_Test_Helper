from code import interact
from functools import total_ordering
import pytest
from typing import Tuple, Union, List
from case_generator import Param_Wrapper, prune_sets, set_generator, wraps_param_vars, wrap_obj
from random import randint
import numpy as np

def create_non_uniform_2D_list(dims: tuple) -> List[List[int]]:
    D2_list = [None] * len(dims) #efficient list creation
    for idx, dim_size in enumerate(dims):
        D2_list[idx] = np.random.random(dim_size)
    
    return D2_list
    
def get_val_idx(D1_idx: int, dims: tuple) -> int:
    """Takes tuple of ints corresponding to some non-uniform 2d-list dimensions and a val representing mapped idx 
    in 1D. Returns the x dimension in given 2D domain"""
    idx = 0
    new_val = D1_idx
    for dim_size in dims:
        new_val -= dim_size
        if new_val < 0: return idx
        idx += 1
    
    return idx

def prod(vals) -> Union[int, float]:
    """Takes a iterable of float or ints and returns product of all elements in iterator"""
    prod  = 1
    for val in vals:
        prod *= val
    
    return prod

@staticmethod
@pytest.mark.parametrize("obj_to_wrap, exp_restrictions", [
        ("hi", []), #normal object 
        (["hi",  "bye"], []), #list len = 2 but no restrictions
        (["hi", ["not", "valid"]], []) ,  #restriction is proper list but not proper type of length
        (["hi", [(5,3,1), (7, 3, 1, 2)]], []),   #restriction is proper list, proper type but not proper length
        (["hi", [(5,3,1), (7, 1, 2)]], [(5,3,1), (7,1,2)]),   #valid restriction list
        (("hi", [(5,3,1), (7, 1, 2)]), [(5,3,1), (7,1,2)])])  #valid restriction tuple
def test_wrap_obj(obj_to_wrap, exp_restrictions: list):
    wrapped_obj = wrap_obj(obj_to_wrap)
    restrictions =  wrapped_obj.restrictions
    assert restrictions == exp_restrictions, "Expected {0} restriction values but found {1}".format(exp_restrictions, restrictions)

def test_set_dim_converter():
    for _ in range(20): #lets run this test 20 times with random values
        idx1, idx2 = randint(2, 10), randint(2, 10)
        D2_list = np.random.random((idx1, idx2))
        x_cord, y_cord = randint(0, idx1 - 1), randint(0, idx2 - 1) #generate random valid index we will check
        str_val = str(x_cord * idx2 + y_cord)
        val  = Param_Wrapper.set_dim_converter(str_val, D2_list)
        assert  val == D2_list[x_cord][y_cord], "bad val expected {0} but found {1}".format(D2_list[x_cord][y_cord], val)

@staticmethod
@pytest.mark.parametrize("param_vars_dims", [
        ((2,3, 7)), #just test a few random list dims -> may want to replace with random dim generator soon
        ((3,4, 4)),
        ((5,3,2,3)),
        ((5,7))])
def test_set_generator(param_vars_dims: tuple):
    D2_list = create_non_uniform_2D_list(param_vars_dims)
    D2_list = wraps_param_vars(D2_list) #create param_wrappers for each element in lists
    num_unique_vals = sum(param_vars_dims)

    #we will keep track of occurrences of each set element
    #as we can calculate the number of expected occurrences easily this gives us a good test 
    #to see if set_generator is giving us correct results
    unique_vals = {str(val): 0 for val in range(num_unique_vals)} 
    sets =  set_generator(D2_list)
    total_combos = prod(param_vars_dims)

    #get occurrences
    for n_set in sets:

        for key in unique_vals:
            if key in n_set:
                unique_vals[key] += 1

    for key, val in unique_vals.items():
        key_val = int(key)
        elm_idx = get_val_idx(key_val, param_vars_dims)
        expected_val = total_combos / param_vars_dims[elm_idx] #expected occurrences of this element

        assert expected_val == val, "Expected element {0} to have {1} instances but found {2}".format(key, expected_val, val)

@staticmethod
@pytest.mark.parametrize("restrictions, param_vars_dims, param_idx, exp_sets", [
    ([(0, 0, 1)], (3,3,3),(1,1), 21), #include 1 val 
    ([(0, 0, 0)], (3,3,3),(1,1), 24), #exclude 1 val
    ([(0, 0, 1), (2, 1, 0)], (3,3,3),(1,1), 20), #include 1 val, exclude 1 val
    ([(0, 0, 1), (2, 1, 1)], (3,3,3),(1,1), 19), #include multiple vals
    ([(0, 0, 0), (2, 1, 0)], (3,3,3),(1,1), 22) #exclude multiple vals 
])
def test_prune_sets(restrictions: List[Tuple[int, int, int]], param_vars_dims: Tuple[int, int, int], param_idx: Tuple[int, int], exp_sets: int):
    """This should cover all legal_set test as well as it will fail is legal_set gives bad results"""
    D2_list = create_non_uniform_2D_list(param_vars_dims)
    D2_list = wraps_param_vars(D2_list) #create param_wrappers for each element in lists
    x_idx, y_idx = param_idx
    wrapped_obj = D2_list[x_idx][y_idx]
    wrapped_obj.restrictions = restrictions
    sets = set_generator(D2_list)
    pruned_sets = prune_sets(D2_list, sets)
    assert len(pruned_sets) == exp_sets, "Expected {0} sets but found {1} sets".format(exp_sets, len(pruned_sets))