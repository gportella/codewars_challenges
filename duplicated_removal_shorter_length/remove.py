#! /usr/bin/env python

# DESCRIPTION:
# Given a string
# s
# s composed of lowercase letters, you can perform the following operation on the string:
# Choose any two characters in the string that are the same and delete them
# Find the minimum lexigraphical string that occurs out of all possibilities after performing this operation on
# s as many times as possible.
# Example
# solve("czaccb") -> "czab"
# solve("geeegfffgdadd") -> "efgad"
# solve("cdeddaeeb") -> "cdaeb"
from collections import Counter

tests = [
    ("czaccb", "czab"),
    ("cazccb", "azcb"),
    ("geeegfffgdadd", "efgad"),
    ("cdeddaeeb", "cdaeb"),
]

tests += [
    ("aaaabbbbdab", "abd"),
    ("aahabbgedbc", "ahbgedc"),
    ("bbfdfbffbcfbh", "bdcfh"),
    ("aggecgdbeef", "acgdbef"),
    ("bcddeffgabbccdeefgg", "abcdefg"),
    (
        "uxpjgfzkgwcrorbxlnyvddzvbjgknjpukuwklcwofgyfvsfpapapgxwoyagnizvfuhdiwdualrzgjolimvgvwjxynwvmyejeycnrpibyizgbbabizhtuegatfnssfsjuieaorpuqjppahrpjhldusnhjhrrgodpleaaxtgplqltzdagtqsecvxiueyvjvlxgcputoalpdfwsstooajiqilpukmizlzpvoxaexrfptuansxrfhzttextiputymglhwxwwbajojmiqwdhnutslgboeutqrxmnfkvisdknohlhdfylqmrekaovrzejdciuqvesuzkwdfdchfjgbyedntgltdkzoeopdaqnioubddfehwrwfafkfqxwxnaurwabnbtfrjcmxrfffqmlxmvubwmyxlvidhvkapmmxozxctwzlxagtvihuaiqiulidaxenztgcfnfbveexlbydhrifotamotahxlhmyqsfbpovxmyvcgbfftbjsoekdymecujbianwfmufslnrkywfdcundnvnfomylmbdglvxytadjplzutehtzlgarkvucaoynlutbrdhqqlfaoqlsdztqlghauejnblktswwstkgcjwhkodthnrewtlauwvwjwwobelshyzgsisvvsimsgkuqznbjzblodpkounmoeejkwapiawnxhdwpwwofupjjmwhqijoccdunurztxijxdqwuwkfxxtmbxqitjfyudfxehmaecmuunrhlpnhopifjfnjijeyiwrfyvophdrpwnuuwufeiinzmmdnogqdxzxldgeqowirpmrcbdzkpkuesqzsdlhiyfcafyfeogfaiftvtnytceebnaakmrudyobpwmubyktgmxtwykqzqoomzzcjzjjexqoddhaylcgxlmrnaexgcxxcsdnflnsqalchwuymqzyqmrdzhjwancombqwkpjviznovehlnpzhcdliguluwgzchktjlz",
        "abcdefghijklmnopqrstuvwxyz",
    ),
]


def trivial_rm(string) -> str:
    rm = [c for c, v in Counter(string).items() if v % 2 == 0]
    return "".join([c for c in string if c not in rm])


def fun(string):
    counts = Counter(string)
    stack = []
    for w in string:
        counts[w] -= 1
        if len(stack) == 0:
            stack.append(w)
            # continue
        while len(stack) and stack[-1] > w and counts[stack[-1]] > 0 and w not in stack:
            stack = stack[:-1]
            if len(stack) == 0 or stack[-1] < w:
                stack.append(w)
        if stack[-1] < w and w not in stack:
            stack.append(w)
            # continue
        elif w not in stack:
            stack.append(w)
            # continue

    return "".join(stack)


def solve(string: str):
    string = trivial_rm(string)
    answer = fun(string)
    print(f"Q: {string} A: {answer}")
    print("-----------")
    return answer


def test_all():
    for k, v in tests:
        assert solve(k) == v


test_all()

# solve(tests[1][0])
