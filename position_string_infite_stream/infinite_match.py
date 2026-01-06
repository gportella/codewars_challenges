#! /usr/bin/env python
"""
Find the first occurence of num in an infinite digit
made up of consecutive/sequential increments, e.g.

12345678910111213
ie
1|2|3|4|5|6|7|8|9|10|11|12|13...

bit tricky... some day.
"""


def find_positions(num):
    n = len(num)
    best_full = None
    best_partial = None

    for width in range(1, n + 1):
        head = num[:width]

        if len(head) > 1 and head[0] == "0":
            continue

        current = int(head)
        idx = width
        next_val = current + 1
        matched = False

        while idx < n:
            expected = str(next_val)
            if num.startswith(expected, idx):
                idx += len(expected)
                next_val += 1
                matched = True
            else:
                break

        remainder = num[idx:]

        if not remainder and matched:
            candidate = (len(head), int(head), head)
            if best_full is None or candidate < best_full:
                best_full = candidate
        elif remainder and matched and remainder[0] != "0":
            prefix = num[:idx]
            candidate = (-idx, int(prefix), prefix, remainder)
            if best_partial is None or candidate < best_partial:
                best_partial = candidate

    if best_full:
        return best_full[2]

    if best_partial:
        _, _, prefix, remainder = best_partial
        return f"{prefix} {remainder}"

    return num


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect the smallest contiguous increasing chunk within a digit string."
    )
    parser.add_argument(
        "digits",
        nargs="?",
        help="Digit string to analyze for contiguous increasing chunks.",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run a few sample digit strings to show expected behavior.",
    )

    args = parser.parse_args(argv)

    if not args.examples and not args.digits:
        parser.print_help()
        return 0

    if args.digits:
        print(find_positions(args.digits))

    if args.examples:
        samples = [
            "456",
            "454",
            "455",
            "910",
            "9100",
            "99100",
            "00101",
            "121314",
            "120121",
            "1001",
        ]
        for sample in samples:
            result = find_positions(sample)
            print(f"{sample} -> {result}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
