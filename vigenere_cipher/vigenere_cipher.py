#! /usr/bin/env python
"""
Collection of code for 3 Katas in Codewars.
"""

from math import gcd
from rich.pretty import pprint
import numpy as np
import re
from collections import deque, Counter
from random import choice

alphabet = "abcdefghijklmnopqrstuvwxyz"
# would use chr and ord, but can't guarantee we don't get a funky alphabet
abc_mapper = {x: pos for pos, x in enumerate(alphabet)}
abc_unmapper = {pos: x for pos, x in enumerate(alphabet)}


def rot_caeser(word: str, pos: int, len_alpha):
    return "".join(
        [abc_unmapper[(abc_mapper.get(char, 0) + pos) % len_alpha] for char in word]
    )


def vigenere(word, key, alphabet, encode=1):
    target_ln = len(word)
    filled = (key * (target_ln // len(key) + 1))[:target_ln]
    print(word)
    print(filled)
    encoded = []
    for w, c in zip(word, filled):
        if w in abc_mapper:
            rot_c = encode * abc_mapper[c]
            enc = rot_caeser(w, rot_c, len(alphabet))
        else:
            enc = w
        encoded.append(enc)
    return f"{''.join(encoded)}"


# this is for the submission, they wanted it in a class...
class VigenereCipher(object):
    def __init__(self, key, alphabet):
        self.alphabet = alphabet
        self.key = key
        # would use chr and ord, but can't guarantee we don't get a funky alphabet

        self.abc_mapper = {x: pos for pos, x in enumerate(self.alphabet)}
        self.abc_unmapper = {pos: x for pos, x in enumerate(self.alphabet)}

    def rot_caeser(self, word: str, pos: int, len_alpha):
        return "".join(
            [
                self.abc_unmapper[(self.abc_mapper.get(char, 0) + pos) % len_alpha]
                for char in word
            ]
        )

    def vigenere(self, word, encode=1):
        target_ln = len(word)
        filled = (self.key * (target_ln // len(self.key) + 1))[:target_ln]
        encoded = []
        for w, c in zip(word, filled):
            if w in self.abc_mapper:
                rot_c = encode * self.abc_mapper[c]
                enc = self.rot_caeser(w, rot_c, len(self.alphabet))
            else:
                enc = w
            encoded.append(enc)
        return f"{''.join(encoded)}"

    def encode(self, text):
        return self.vigenere(text)

    def decode(self, text):
        return self.vigenere(text, encode=-1)


#  The rest if for breaking the cipher


def maximumOccurringString(s, K):
    M = {}
    D = deque()
    for i in range(K):
        D.append(s[i])
    M[str("".join(list(D)))] = M.get(str("".join(list(D))), 0) + 1
    D.popleft()
    for j in range(i, len(s)):
        D.append(s[j])
        M[str("".join(list(D)))] = M.get(str("".join(list(D))), 0) + 1
        D.popleft()

    ss = sorted(M.items(), key=lambda x: x[1], reverse=True)
    for good, count_good in ss:
        yield good, count_good


def ioc(cf, k):
    if k == 0:
        return 0
    splits = [cf[c::k] for c in range(k)]
    ics = 0
    for column in splits:
        counts = np.array(list(Counter(column).values()))
        ic = 26 * sum(counts * (counts - 1)) / (len(column) * len(column) - 1)
        ics += ic

    return float(ics / k)


def friedman(cf, max_key=20):
    scores = {}
    for k in range(max_key, 1, -1):
        score = ioc(cf, k)
        scores[k] = score

    if len(scores):
        av_sc = {}
        av_score = sum([sc for sc in scores.values()]) / len(scores)
        for k, sc in scores.items():
            av_sc[k] = (sc) ** 2 / av_score
        winners = sorted(av_sc.items(), key=lambda x: x[1])
        print(winners)
        return winners[-1][0]
    else:
        return 1


def get_key_length(cf, max_key=0, max_k=8, min_occ=3):
    possible = []
    for K in range(2, max_k):
        for sub, n_times in maximumOccurringString(cf, K):
            if sub and n_times >= min_occ:
                ha = np.array(
                    [match.start() for match in re.finditer(re.escape(sub), cf)]
                )
                diff_pos = ha[1:] - ha[:-1]
                part_gcd = gcd(*list(diff_pos))
                if part_gcd > 1:
                    possible.append(part_gcd)

    counts_possible = Counter(possible)
    if counts_possible:
        my_guess = next(iter(counts_possible.most_common(1)))
        return my_guess[0]
    else:
        print("Default to Friedman")
        return friedman(cf, max_key=max_key)


key_secret = "password"
sample = """ Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam justo diam, tempus a sagittis eget, sodales ac neque. Curabitur sodales arcu urna, nec cursus massa placerat id. Morbi rutrum libero at nibh egestas interdum. Duis turpis urna, elementum non quam eget, gravida blandit risus. Aliquam erat volutpat. Curabitur pellentesque, ex et ultrices sollicitudin, magna ante imperdiet nunc, sed tempus sem orci et risus. Donec porttitor velit quis enim egestas, vel venenatis risus hendrerit. Praesent semper a mauris ut malesuada. Fusce ornare nisi vitae elit pharetra, facilisis viverra velit semper.
Etiam ullamcorper ornare posuere. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aenean non odio dolor. Cras nec dolor nisl. Pellentesque eu mauris finibus, aliquam mauris sed, tempor diam. Proin rhoncus purus imperdiet lobortis fringilla. Maecenas pretium eros nec maximus vulputate. Praesent interdum, ante eget pharetra commodo, enim purus tempus sapien, ut congue risus massa ut ipsum. In hac habitasse platea dictumst. Vivamus lectus urna, tempor vel orci et, sodales tristique massa.
Duis tempus, nisi sit amet laoreet fermentum, ex tellus venenatis neque, vel commodo ligula eros eget justo. Quisque suscipit diam dolor, quis vulputate dolor commodo ac. Morbi mauris eros, pharetra eget dapibus et, tincidunt eu diam. Mauris bibendum velit quis pulvinar condimentum. Donec tristique magna at eleifend euismod. Donec vitae turpis a nisi fermentum commodo non eu lacus. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam vel lectus quis quam imperdiet consequat at at nisi.
Donec pharetra varius dapibus. Duis molestie erat nibh, blandit lacinia sapien iaculis et. Mauris vel laoreet diam. Mauris consequat non purus sit amet luctus. Sed accumsan tristique dolor vel lobortis. Suspendisse potenti. Vestibulum pulvinar posuere risus, ac ullamcorper nisi consectetur nec. Sed vulputate sagittis tellus, at gravida nisl molestie a. Suspendisse non massa leo. In at ligula eu urna ultricies suscipit. In non euismod dolor, quis malesuada leo. Aenean imperdiet blandit ante rhoncus finibus.
Praesent finibus volutpat est, et rhoncus nulla ultrices et. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam non nibh lacinia, mollis felis ac, ultrices velit. Nunc posuere pharetra nibh, nec ullamcorper metus sagittis vitae. Nunc et magna id massa eleifend semper eget sit amet mauris. Nunc est eros, condimentum quis luctus id, tincidunt id orci. Nunc a faucibus justo, in semper massa. Proin gravida tincidunt dui vel sagittis. Pellentesque mi odio, suscipit id enim ac, accumsan tincidunt est. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Cras leo nibh, dictum id libero nec, finibus pretium odio. Duis sapien neque, gravida a mauris sit amet, efficitur aliquet ex. Praesent tristique velit a risus ultricies, id pellentesque odio suscipit. Ut auctor mollis dui sit amet tempor"""

sample = """
As they rounded a bend in the path that ran beside the river, Lara recognized the silhouette of a fig tree atop a nearby hill. The weather was hot and the days were long. The fig tree was in full leaf, but not yet bearing fruit.
Soon Lara spotted other landmarks—an outcropping of limestone beside the path that had a silhouette like a man’s face, a marshy spot beside the river where the waterfowl were easily startled, a tall tree that looked like a man with his arms upraised. They were drawing near to the place where there was an island in the river. The island was a good spot to make camp. They would sleep on the island tonight.
Lara had been back and forth along the river path many times in her short life. Her people had not created the path—it had always been there, like the river—but their deerskin-shod feet and the wooden wheels of their handcarts kept the path well worn. Lara’s people were salt traders, and their livelihood took them on a continual journey.
At the mouth of the river, the little group of half a dozen intermingled families gathered salt from the great salt beds beside the sea. They groomed and sifted the salt and loaded it into handcarts. When the carts were full, most of the group would stay behind, taking shelter amid rocks and simple lean-tos, while a band of fifteen or so of the heartier members set out on the path that ran alongside the river.
With their precious cargo of salt, the travelers crossed the coastal lowlands and traveled toward the mountains. But Lara’s people never reached the mountaintops; they traveled only as far as the foothills. Many people lived in the forests and grassy meadows of the foothills, gathered in small villages. In return for salt, these people would give Lara’s people dried meat, animal skins, cloth spun from wool, clay pots, needles and scraping tools carved from bone, and little toys made of wood.
Their bartering done, Lara and her people would travel back down the river path to the sea. The cycle would begin again.
It had always been like this. Lara knew no other life. She traveled back and forth, up and down the river path. No single place was home. She liked the seaside, where there was always fish to eat, and the gentle lapping of the waves lulled her to sleep at night. She was less fond of the foothills, where the path grew steep, the nights could be cold, and views of great distances made her dizzy. She felt uneasy in the villages, and was often shy around strangers. The path itself was where she felt most at home. She loved the smell of the river on a hot day, and the croaking of frogs at night. Vines grew amid the lush foliage along the river, with berries that were good to eat. Even on the hottest day, sundown brought a cool breeze off the water, which sighed and sang amid the reeds and tall grasses.
Of all the places along the path, the area they were approaching, with the island in the river, was Lara’s favorite.
The terrain along this stretch of the river was mostly flat, but in the immediate vicinity of the island, the land on the sunrise side was like a rumpled cloth, with hills and ridges and valleys. Among Lara’s people, there was a wooden baby’s crib, suitable for strapping to a cart, that had been passed down for generations. The island was shaped like that crib, longer than it was wide and pointed at the upriver end, where the flow had eroded both banks. The island was like a crib, and the group of hills on the sunrise side of the river were like old women mantled in heavy cloaks gathered to have a look at the baby in the crib—that was how Lara’s father had once described the lay of the land.
Larth spoke like that all the time, conjuring images of giants and monsters in the landscape. He could perceive the spirits, called numina, that dwelled in rocks and trees. Sometimes he could speak to them and hear what they had to say. The river was his oldest friend and told him where the fishing would be best. From whispers in the wind he could foretell the next day’s weather. Because of such skills, Larth was the leader of the group.
“We’re close to the island, aren’t we, Papa?” said Lara.
“How did you know?”
“The hills. First we start to see the hills, off to the right. The hills grow bigger. And just before we come to the island, we can see the silhouette of that fig tree up there, along the crest of that hill.”
“Good girl!” said Larth, proud of his daughter’s memory and powers of observation. He was a strong, handsome man with flecks of gray in his black beard. His wife had borne several children, but all had died very young except Lara, the last, whom his wife had died bearing. Lara was very precious to him. Like her mother, she had golden hair. Now that she had reached the age of childbearing, Lara was beginning to display the fullness of a woman’s hips and breasts. It was Larth’s greatest wish that he might live to see his own grandchildren. Not every man lived that long, but Larth was hopeful. He had been healthy all his life, partly, he believed, because he had always been careful to show respect to the numina he encountered on his journeys.
Respecting the numina was important. The numen of the river could suck a man under and drown him. The numen of a tree could trip a man with its roots, or drop a rotten branch on his head. Rocks could give way underfoot, chuckling with amusement at their own treachery. Even the sky, with a roar of fury, sometimes sent down fingers of fire that could roast a man like a rabbit on a spit, or worse, leave him alive but robbed of his senses. Larth had heard that the earth itself could open and swallow a man; though he had never actually seen such a thing, he nevertheless performed a ritual each morning, asking the earth’s permission before he went striding across it.
“There’s something so special about this place,” said Lara, gazing at the sparkling river to her left and then at the rocky, tree-spotted hills ahead and to her right. “How was it made? Who made it?”
Larth frowned. The question made no sense to him. A place was never made, it simply was. Small features might change over time. Uprooted by a storm, a tree might fall into the river. A boulder might decide to tumble down the hillside. The numina that animated all things went about reshaping the landscape from day to day, but the essential things never changed, and had always existed: the river, the hills, the sky, the sun, the sea, the salt beds at the mouth of the river.
He was trying to think of some way to express these thoughts to Lara, when a deer, drinking at the river, was startled by their approach. The deer bolted up the brushy bank and onto the path. Instead of running to safety, the creature stood and stared at them. As clearly as if the animal had whispered aloud, Larth heard the words “Eat me.” The deer was offering herself.
"""


encoded_sample = r"""?W\JLz>*]G&?4DX
'4IX,,K"?nlP$;4G	],VW}xlU	;[s<.]Iy?HDT
=8F_&[N-!CU'	-6}X6^VS>4B*0'eIX&]Fy{kUW\,0K@(>C"!MIP+4JX4~V+]nHT${eu>\^PyBVF,`<[T.[{QZ`=lR>+bv;P}JS!&JV\-4I<P.K$\oSz1-1I<&@CP`olX J7z~P#AK/4QX\"4bxc=Gy4vQW>!4KX	>RV?B	z`-[rX(=KZ>BF#D<[D.,,"W|o2zf-[i-W)1ymmJT =8w@(V#!?BJR>-[u<\.TW<oEz0'4bm.<G"?BFz@(fy<[VC'!;J"
.iJ@'[Gy}pl)ZdJ]&}K#|;2
_(ib}*_W(;DJ$ Jmr~P]Q(!nF(\;lv;wV%V;BMTJrr/',IS!sTz|-eN_P}Qy\kWT$!hF\*]mO!FB'`ZdKX>;m(\olR`/7v}P,Uy?kS!5J0JXW*xIK4I$3$lv}uVJS!nJS jjb{^-NW
rlW`<[N`[@3];CJ(|([v_]>TS`IlQ.avX]=Gy=sQW\;[r_)VR)<vJ(_$3b -:m(?mI#`:kvX.]m(\olG==7b:*]V)	I2zb?4EX':H#	ol)_(i{X]=Q)[r0z.cvX\@KZ`oEz@;oG &]CZ2CU($"eL])VQQ=kTX	-0C]|VD&?kLz0'4b:._JS	4J#$=7vXW(VV!mF#0>hPyf>MW~oEX>l 2"
"""

encoded_sample_1 = r"""]"24;UW!mwR)
%>.uJ,?qvU$!ccAOu'_=GF.6yIwV+qcB)l:<Ir#W2zOU%*9DL{
?}:BS!"4)!*!ccA]w*^:7F(14HM*$dEA\l%>YAp[5kMXU/~$U\z+/%rp\cnB%%*~^,)z+|IPF,fyGXcGbsL:h&V<uJEJsBQ$!qwk'l<=#Ep/blMQQ'9tL)
+]IuN-?btIOGoAE'hMv5uJEovKTQXdLk l<=#EE6xfMAYgAL(u']iFp'qv,Y*d~k0qM#XeCj?#xUU*sAF-fM+rW#0kIwT!ruR-e'/IGMY?&DSU*dJEOf+_&rWECf6Y)oGS[l$[#7TZ?DMM$/ksT-r:vp'M#g4MQ&;ssT-r:V?nXEbyOwT!rwR^h&CIPMUfvz%A\9tB%j'V'Fp%byRZA:ncH%y'VZET%2xfMA<9JI%q@V-sp.5ofOY,gwROd?V#nW&m4v%AN7nuy
*^?r!Yf@fTUGcAD<*@V.HG&6CCwX%rcW=u-C
nX#guDwU*sAR)o_VZET%24OTUGbAP,h>VYAIEdEwXY/gwDOw*:IGJW5xD#+!~ANOw*:IeE.54xQ$:tJYv
  #ApV2pJ$UGszI[/M}&BZ!5@f%%)dcS/l.[#qpWfIK&Q*9DY[w?V!BZ&14JOSWrAO<d.[[7G,2kFw*$dcC-s*::7N(?DCQAN5LHOf']<HW=V'DWY,dvI%,"""


def test(sample_text, epochs=5):
    print(f"For text of sample lenght {len(sample_text)}")
    solution_grid = {}
    for max_k in range(3, 8):
        solution_grid[max_k] = {}
        for min_occ in range(2, 6):
            solution_grid[max_k][min_occ] = 0
            largest_len = []
            for _ in range(epochs):
                for pwd_len in range(2, 20):
                    key_secret = "".join(choice(alphabet) for _ in range(pwd_len))
                    vg = VigenereCipher(key=key_secret, alphabet=alphabet)
                    cf = vg.encode(sample_text)
                    guessed = get_key_length(
                        cf, max_key=20, max_k=max_k, min_occ=min_occ
                    )
                    if guessed != pwd_len:
                        largest_len.append(pwd_len)
                        break
            if len(largest_len):
                solution_grid[max_k][min_occ] = sum(largest_len) / len(largest_len)

    pprint(solution_grid)


expected_freq = {
    "E": 0.1202,
    "T": 0.0910,
    "A": 0.0812,
    "O": 0.0768,
    "I": 0.0731,
    "N": 0.0695,
    "S": 0.0628,
    "R": 0.0602,
    "H": 0.0592,
    "D": 0.0432,
    "L": 0.0398,
    "U": 0.0288,
    "C": 0.0271,
    "M": 0.0261,
    "F": 0.0230,
    "Y": 0.0211,
    "W": 0.0209,
    "G": 0.0203,
    "P": 0.0182,
    "B": 0.0149,
    "V": 0.0111,
    "K": 0.0069,
    "X": 0.0017,
    "Q": 0.0011,
    "J": 0.0010,
    "Z": 0.0007,
}


ASCII_CAP_OFFSET = 65
LEN_ALPHABET = 26


def dechipher(cf, key_len):
    splits = [cf[c::key_len] for c in range(key_len)]

    the_key = []
    for column in splits:
        if not len(column):
            break
        freq_counts = {k: v / len(column) for k, v in Counter(column).items()}
        most_common = [
            x for x in sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        diff_decide = 1
        shifter = 0
        # I'm going to take the most frequent one as the starting point
        # try to match it to E, and get the difference wrt expected distribution
        # of all the frequencies
        for c, _ in most_common[: len(expected_freq) - 1]:
            offset_m = ord(c) - ord("E")
            maybe_freq = {
                chr(
                    ((ord(cc) + offset_m - ASCII_CAP_OFFSET) % LEN_ALPHABET)
                    + ASCII_CAP_OFFSET
                ): vv
                for cc, vv in expected_freq.items()
            }
            diff_expr = 0
            for cc, vv in maybe_freq.items():
                if cc in freq_counts:
                    diff_expr += (freq_counts[cc] - vv) ** 2
            if diff_expr < diff_decide:
                shifter = offset_m
                diff_decide = diff_expr
        w = chr(shifter % LEN_ALPHABET + ASCII_CAP_OFFSET)
        the_key.append(w)

    the_key = "".join(the_key)
    return the_key


if __name__ == "__main__":
    key = "CODEWARS"
    ciphertext = "NSWXARWJGEXIJCZWUZLOAWFJFTUIMUVFEWHWPEEVVCYENYSGVVECSRZLGFDRZBPKWPMIYTFFGQDRJOKOTWWIWNVKUOBEXOLLZFDCOWZLJCXXQSZFITUIMUVFVLVEJDKZGSVWWYNANZKERERFKRLSOYEUTOWMYLVLVSUJNEHMGBFCEFKZGSVWWYZKCPRYPTYWHFHUQEELWGHSBXISAGWSPRVSVNHFNAJAPEDXWRUAHTHVANKSWHKSNSYSXSKEXIKKYVLGDCRFDSUIBLVUVSGMJTYWKFXWAOWDGHWINSYWOWQKSAPKYFLXENXKVMOIBOIWZOPTHEZKXWVMXLPVKTIINEELHFRQBALDMBHVOLVLVSUFEGISOHUMCRREYCUHBRVIWSQGEEJOQFGPANXLJOQHOEELGBFIHEEYVVFEJBVUCZFYHAKWFTRVOPVUKTLGWUKZQFVEJDLKGRWSLRFNGCUHESGJQJHEQTYGTGKMLOWLGLWWAVVFHCUEQTYGTGZLKSVKVMOIOAIWPCWWKDZNGFJIJTRUEIUEPERNGFDKALVLVSUJNEHMGBFMASTSPCQPUBVYNSDRADSQCBDPUZZFIOOENGVSOCXRPOWJGDUIOEELCHLZATVPVKLXDTYWCJDMHASANWWCKFDGFSURYODHWHLRCAEVECOPACKAQBVSBLRJISWITTTGTDRVWSLUJQDPYUCSVWRROAIWGOVMHYDSFSHBWMGDGGFEJBVVTOZRBRFECJDVEEKQQTVSQRTWUDUIOSIWRCUXENXJGZLKEOLKVSAXOSTAGBWMBITLGLWWWNUYGBHVWLWAEHLSJAEVVVHVAAIWFWIJARVFESVIOPVUKOOPUFFJISQINACXKQWMKNNAVVWLAPFKKHLSJOWZCBGMSIKZJPHGKMZFIARVACFEOCQLARSWTHVDEMZFJWVGHAJKKQLRPRFVWQWSNYTJADWSCRRHJMWITTTGFSVEJDJWEFHXSRZLKBJKEVVKVVHIJGCAUVOIPTVJHFHUQEEUAGHUQEEUGOVIPAFFTWVLZLWUOIJCLWSNMXAUVTYWOCVXYODEQBOIPTVJROLVOAJLJVHEJRVWTWQSJAKFFGWIOEEGHHHIZOILKVLEOTFSPRWLAMFKVQRQIOEVQIEPADCWVHHVOAJDNSHWOOFLVTIVNNEHRQFXDEKGRHZIHVVDGHWINSTGODUMOERTQIWSBTYWVCWEHUJSISWLATFHGWJLPLVLVSUWYODHTWVIWBFMVCIXDEKGVOOYOAXWNSWXARWJGEXIJCPSUOIYJCKAQBRJNAECEOQFAFZLVSGAALCTAGHZARRDTOQOBUEUVWRROWZLJHKIPWFHCFDQATVJECFLKBVLCFDRGFLFEHLSJBVAPUWLABVKVOQSPHVJTOQOBUEUVWRRSIKZPCDHFUJLCPOIBRVWROUEIEKWTOOWKFZLUHKIHEKLGFIVAQLWPQBHESKJKPXXEOEJGOVSJASDAKHPHTYWUOPIBUEUVWRRDAJTGSQYOEULQTLXPHVSOWQSWCZVHFHUQEEUAWQTNOKWKBVIMUVFESVEOPPMUWQKPHVNKQFMLHVJQFVSIEFLJSUGEPYWTPDWADFFCGWVWDUDKBJGDETCGFESWRULADLGWLCQWGHWWMEWOCQMYSLUJOVEOIELQSUVZRFHRWQKPHVKGQRRZRKGTSPIIBVJVVHXKPVAIVWGDAISEHHVOTYWWGHSBLVLVSUJNEHMGBFMASRFFTUIMUVFEMDRWLPKKGSPWYJSHIQHWMVFVOOVKLVAPQUCLTFYTOPWWNUKGJHVWLNGTRSYVZCWIOPIOIEUNIGMJGYSPUPEJSTJCPEPAAEVVVHXALVNKGLSJGREGGKSSWYWGZRJBOILWBHSJEFXVVHIWRCAGGWHASTJKDWMKNZFEZDWOITSNZLXARRLWFHSBAGHNMLRCTYWMBRAHEUYGCIIJGCAUVOIPTVJHFHUQEEUAHRWKLMAPUDGNYGLQUUEIIJXQIQHENVSRCHWBADGWGVXKRPLJSJSHDSMIKKINEKZGAHXDOUAUGXGYEJKHIOPUAGHNWHHPOUWEWSLARREGGVECEZFUHUYYTZFICQXDENZGFHEXOLLUCIEPRVSUIUIDIUVGBECYAGLCWQOEDUDGHWINFIWSIHRYIVKJOGEOTIGPUHJBETLQBWLADVKKUQSBSFEGYHCXORJFZDCKUKKVVHQKSKXTSTYANKDGHWINSRJGCQXDESGVHRQNONGHHKIXLZUMSQWZEIXGFWCLENJKHHVWNULJSKSIEIGYCIXDEUNQFDOOIDHNWIMADBWAPREND"
    answer = dechipher(ciphertext, len(key))
    print(answer)
