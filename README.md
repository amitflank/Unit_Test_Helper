This Project was designed with pytest unit testing in mind but can probably be adapted with some minimal effort.

This project seeks to make creating large numbers of different test conditions as painless as possible. We do this by allowing you
to specify several potential values for a argument and it's relationships with other potential arguments you pass.

## Generating Arguments  

Lets say I would like to create 27 combinations of 3 strings each which may contain 3 unique values. Lets look at how we would specify this.

`gen_param_data = [["hi", "bye", "dude"],
["no", "way", "jose"],
["good", "time", "guy"]]`

Here each outer list represents possible arguments for the given outer list index. So the first value for each of our 27 combinations will be **"hi"**, **"bye"** or **"dude"**.

First we need to wrap our values in a special `Param_Wrapper`. We can do this by using `wrapped_values = wraps_param_vars(gen_param_data)`

We can now use this to generate are argument combinations.
`combos = generate_params(wrapped_values)`

This will output a list of 27 tuples each of which contains a unique combination of each of our words sets that can be passed to a say pytest parameterization. 

### **Adding restrictions to arguments**
This is nice but lets say I have some restriction on the relationships of my word sets, say if I pass **"hi"** and **"way"** the function im testing will break or my tests take a long time and I don't care about any combination of those two values.

I can do this by adding restrictions to **"hi"**. 

`restricted_hi =  ("hi", [(1,1,0)])`<br />
here the first two values represents the indexes at which the related value is located and the final value represents the relationship where 1 indicates value **must** be paired and 0 indicates it **cannot** be paired.

So lets replace **"hi"** with `restricted_hi`

`gen_param_data = [[restricted_hi, "bye", "dude"],
["no", "way", "jose"],
["good", "time", "guy"]]`

Now if we run: `combos = generate_params(wrapped_values)`

We find combos no longer includes the following 3 tuples: <br />
**("hi", "way", "good")** <br />
**("hi", "way", "time")** <br />
**("hi", "way", "guy")**

We could force **"hi"** to only appear in tuples that include **"way"** by changing our 0 to a 1 which would remove 6 values. 

Additionally we can have as many restrictions as we want. Lets say we wanted **"hi"** to exclude **"way"** but include **"good"**. We can simple add that restriction to `restricted_hi`.

`restricted_hi =  ("hi", [(1,1,0), (2,0,1)])`<br/><br/>

## Functions
Lets make a simple function:

`def add_num(val1, val2):`<br/>
&emsp; `return val1 + val2`<br/>

Now lets say we want to pass it 4 potential values, 2 for each argument. We can do this the same way we did previously however we will now use `Fxn_Wrapper`.

`my_args = [[1,2], [5,7]]` <br/>
`my_fxn_wrapper = Fxn_Wrapper(add_num, my_args)`

>**Note**: We don't need to manually wrap our args this time as Fxn_Wrapper will do it for us. 

We can use the `evaluate_fxn` to get our output:<br/>
`print(my_fxn_wrapper.evaluate_fxn())`
>[6, 8, 7, 9]

### **Functions with restrictions** 

We can also add restrictions to the function values we pass, just like we did with our initial argument generator. 

Lets make sure our **1** only gets added to **7** <br/>

`restricted_1 = (1, [(1, 0, 1)])`<br/>
`my_args = [[restricted_1, 2],`<br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;`[5, 7]]` <br/>
`my_fxn_wrapper = Fxn_Wrapper(add_num, my_args)`<br/>
`print(my_fxn_wrapper.evaluate_fxn())`
>[8, 7, 9]

### **Nesting Functions**

We can also nest functions inside other functions or as part of argument generation. 

Lets make a second function:

`def mul_num(num1, num2: int):`<br/>
&emsp;`return val1 * val2`

Let see what happens when we pass a Fxn_Wrapper as a argument to another function wrapper. 

`inner_args = [[1, 2],[3, 4]]`<br/>
`inner_fxn = Fxn_Wrapper(add_num, args)`<br/>
`outer_args = [[1,2], [3, inner_wrap]]`<br/>

When a Fxn_Wrapper receives a Fxn_Wrapper as an argument, it replaces that Fxn_Wrapper with the output of its `evaluate_fxn` call. So this effectively reads:

>outer_args = [[1,2], [3, 4, 5, 5, 6]]

`outer_wrap = Fxn_Wrapper(mul_num, outer_args)`<br/>
`print(outer_wrap.evaluate_fxn())`
>[3, 4, 5, 5, 6, 6, 8, 10, 10, 12]

There is no restriction on how many functions you can nest, though the number of arguements you generate will increase very rapidly.







    