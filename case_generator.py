import itertools
import inspect
from typing import Any, List, Tuple


class Param_Wrapper():
    """Wraps an argument and it's restrictions. Also includes some useful methods for checking legality
    of this arguments set and for converting between 1D and 2D idx representations.

    Args:
        value: value to wrap
        restrictions: A list of restrictions for this arg in the from of Tuple[int, int, int].
        First two ints represents x,y idx of restricted object and third int represents relationship, 1 included, 0 excluded.
    """

    def __init__(self, value: Any, restrictions: List[Tuple[int, int, int]] = []) -> None:
        self.value = value
        self.restrictions = restrictions #list_idx, set_idx, restriction_type(1 included, 0 excluded)

    @staticmethod
    def convert_1d_idx_to_2d(d1_val: str, D2_list: List[list]) -> Tuple[int, int]:
        """takes 1D mapped value and finds the 2D equivalent and returns idx of that element in passed D2_list."""
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
    def set_dim_converter(d1_val: str, D2_list: List[list]) -> Any:
        """Given mapped 1d idx value and 2D list return corresponding value in 2D list"""
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
                raise ValueError("passed bad restriction value {0}. restrictions must be 0: exclusive or 1: inclusive".format(restriction[2]))
        
        valid_exclude_set = len(exclude_set & p_set) == 0 #make sure we have no elements from exclude set
        return include_set.issubset(p_set) and valid_exclude_set

class Key_Param_Wrapper(Param_Wrapper):
    """Wraps a dictionary argument it's restrictions and its key. Also includes some useful methods for checking legality
    of this arguments set and for converting between 1D and 2D idx representations. Inherits from Param_Wrapper.

    Args:
        value: value to wrap
        restrictions: A list of restrictions for this arg in the from of Tuple[int, int, int].
        First two ints represents x,y idx of restricted object and third int represents relationship, 1 included, 0 excluded.
        key: key associated with this value.
    """

    def __init__(self, value: Any, restrictions: List[Tuple[int, int, int]] = [], key: str = None) -> None:
        super().__init__(value, restrictions)
        self.key = key

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
    """wraps nested list of values in Param_Wrappers so they can be used in our set comparison operations"""
    wrapped_nested_list = [None] * len(unwrapped_vars)
    for i, param_list in enumerate(unwrapped_vars):

        wrapped_list = [None] * len(param_list)
        for j, param in enumerate(param_list):
            wrapped_list[j] = wrap_obj(param)
        
        wrapped_nested_list[i] = wrapped_list
    
    return wrapped_nested_list

def prune_sets(param_vars: List[List[Param_Wrapper]], sets: List[set]):
    """Modifies passed sets by removing any set that contains a param whose restrictions are violated.
    Returns modified sets."""
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

def set_generator(param_vars: List[List[Param_Wrapper]]) -> List[set]:
    """Takes nested list of Param_Wrapper and maps each element to unique value based on it's location in nested List. 
    
    Example if we had a Nested list with dimensions (3,2,3) the value assigned to parameter at idx (1,1) would be 4[3 * 1 + 1],
    while idx (2,1) would be 6 [3*1 + 2*1 + 1].
    
    returns:
        List of sets with mapped values"""
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

def generate_params(param_vars: List[List[Param_Wrapper]]) -> List[tuple]:
    """Takes a nested list of Param_Wrapper where each inner list represent a potential argument at that inner lists idx. 

    returns:
        A List of tuple where each tuple represents a valid set of arguments which do not violate any Param_Wrapper restrictions.
    
    Example:
        We pass param_vars whose Param_Wrapper values are [[1,2] [3,4]] with no restrictions.
        Output would be [(1,3), (1,4), (2,3), (2,4)].
    """
    sets = set_generator(param_vars)
    pruned_sets = prune_sets(param_vars, sets)
    args_list = [None] * len(pruned_sets)

    for idx, set_vals in enumerate(pruned_sets):
        args = ()
        for val in set_vals:
            args += (Param_Wrapper.set_dim_converter(val, param_vars),)
        args_list[idx] = args
    return args_list
    

class Fxn_Wrapper():

    def __init__(self, fxn: callable, args: List[List[Key_Param_Wrapper]], keys: List[str] = None):
        assert callable(fxn), "must pass a function to Fxn_wrapper"
        self.fxn = fxn 
        self.args = args
        
        #extract keys if not passed
        if keys is None:
            self.keys = inspect.getfullargspec(fxn).args
        else:
            self.keys = keys

        assert len(self.args) == len(self.keys), "Number of passed arguments must match number of keys"

    def eval_args(self, args: List[Key_Param_Wrapper]) -> list:
        pass_dict = {key: None for key in self.keys}
            
        for arg in args:
            pass_dict[arg.key] = arg.value
        
        return self.fxn(**pass_dict)


    def evaluate_fxn(self) -> list:
        new_args = generate_params(self.args)
        results = []

        for arg_vals in new_args:
            arg_list = list(arg_vals)
    
            for idx, arg in enumerate(arg_vals):
                #Recursively call ourselves and add all retrieved values as valid arguments at idx
                if type(arg.value) is Fxn_Wrapper:
                    unpacked_args = arg.value.evaluate_fxn()
                    unpacked_args = [Key_Param_Wrapper(u_arg, key = arg.key) for u_arg in unpacked_args]
                    arg_list[idx] = unpacked_args
                   
            for idx, val in enumerate(arg_list):
                arg_list[idx] = [val] if type(val) is not (list) else val #turn non-list elements into lists
            
            results += [self.eval_args(arg) for arg in itertools.product(*arg_list)]
    
        return results