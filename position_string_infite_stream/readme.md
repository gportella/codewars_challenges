# Description:

There is a infinite string. You can imagine it's a combination of numbers from 1 to n, like this:

`"123456789101112131415....n-2n-1n"`

>Please note: the length of the string is infinite. It depends on how long you need it(I can't offer it as a argument, it only exists in your imagination) ;-)

Your task is complete function findPosition that accept a digital string num. Returns the position(index) of the digital string(the first appearance).

For example:

```
findPosition("456") == 3
because "123456789101112131415".indexOf("456") = 3
            ^^^
```

Is it simple? No, It is more difficult than you think ;-)

findPosition("454") = ?
Oh, no! There is no "454" in "123456789101112131415",
so we should return -1?
No, I said, this is a string of infinite length.
We need to increase the length of the string to find "454"


```findPosition("454") == 79```
because 

```
"123456789101112131415...44454647".indexOf("454")=79
                                   ^^^
```
The length of argument num is 2 to 15. So now there are two ways: one is to create a huge own string to find the index position; Or thinking about an algorithm to calculate the index position.

Which way would you choose? ;-)

Some examples:

```
 findPosition("456") == 3
 ("...3456...")
       ^^^
 findPosition("454") == 79
 ("...444546...")
        ^^^
 findPosition("455") == 98
 ("...545556...")
       ^^^
 findPosition("910") == 8
 ("...7891011...")
        ^^^
 findPosition("9100") == 188
 ("...9899100101...")
         ^^^^
 findPosition("99100") == 187
 ("...9899100101...")
        ^^^^^
 findPosition("00101") == 190
 ("...99100101...")
         ^^^^^
 findPosition("001") == 190
 ("...9899100101...")
           ^^^
 findPosition("123456789") == 0
 findPosition("1234567891") == 0
 findPosition("123456798") == 1000000071
 ```

# Clues

Start by observing that any match must be composed of whole decimal pieces from consecutive integers, possibly plus a partial prefix/suffix. Keep track of how many digits you’d get from numbers with 1 digit, 2 digits, etc. This lets you convert “how many numbers until here” into an absolute character offset.
For each plausible alignment, split the target string into slices that could correspond to an actual integer and its neighbors (e.g. consider whether "454" might come from "44|45|46" or wrap between "4|54|5"). Only a small set of candidate starting numbers should remain once you enforce that digits increment correctly.
Once you have a candidate starting integer, you can compute the exact position without building the stream: count the digits taken by all complete numbers before it, add any partial overlap, and adjust for 0-based vs 1-based indexing.
Pay special attention to cases with leading zeros or where the block crosses a digit-count boundary (e.g. "...9899100..."). Those often mean the logical starting number is shorter or longer than the block itself, but the arithmetic for accumulating digit lengths still works.
Thinking in terms of candidate starting numbers and verifying them mathematically saves you from ever needing the “actual” infinite string.
