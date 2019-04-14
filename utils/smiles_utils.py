import re


def parse_smile(canonical_reaction):
    regexp = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\|=|#|-|" \
             "\+|\\\\\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|\.|=)"
    if '>' not in canonical_reaction:
        source_smile = " ".join(re.findall(regexp, canonical_reaction))
    else:
        reactants, reagents = canonical_reaction.split('>')
        reactants = ' '.join(re.findall(regexp, reactants))
        reagents = reagents.split(".")
        reagents = ["A_" + "".join(re.findall(regexp, reagent)) for reagent in reagents]
        source_smile = reactants + ' > ' + " ".join(reagents)
    return source_smile


def get_reaction(source_smile, target_smile):
    if '>' in source_smile:
        source_smile = source_smile.split('>')[0]
    reaction = source_smile.replace(" ", "") + ">>" + target_smile.replace(" ", "")
    return reaction


if __name__ == "__main__":
    print(parse_smile('CS(=O)(=O)Cl.OCCCBr>CCN(CC)CC.CCOCC'))

