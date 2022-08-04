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

This is nice but lets say I have some restriction on the relationships of my word sets, say if I pass **hi** and **way** the function im testing will break or my tests take a long time and I don't care about any combination of those two values.

I can do this by adding restrictions to **hi**. 

`restricted_hi =  ("hi", [(1,1,0)])`<br />
here the first two values represents the indexes at which the related value is located and the final value represents the relationship where 1 indicates value **must** be paired and 0 indicates it **cannot** be paired.

So lets replace **hi** with `restricted_hi`

`gen_param_data = [[restricted_hi, "bye", "dude"],
["no", "way", "jose"],
["good", "time", "guy"]]`

Now if we run: `combos = generate_params(wrapped_values)`

We find combos no longer includes the following 3 tuples: <br />
**("hi", "way", "good")** <br />
**("hi", "way", "time")** <br />
**("hi", "way", "guy")**

We could force **hi** to only appear in tuples that include **way** by changing our 0 to a 1 which would remove 6 values. 

Additionally we can have as many restrictions as we want. Lets say we wanted **hi** to exclude **way** but include **good**. We can simple add that restriction to `restricted_hi`: 

`restricted_hi =  ("hi", [(1,1,0), (2,0,1)])`<br/><br/>

# Functions
Lets make a simple function:

`def add_num(val1, val2):`<br/>
&emsp; `return val1 + val2`<br/>

Now lets say we want to pass it 4 potential values. We can do this the same way we did previously but we will need to specify 
 


    