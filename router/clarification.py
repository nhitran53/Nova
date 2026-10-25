"""Bounded, explicit selection of an already offered BERT candidate (not a first-turn router)."""
import re


def normalized(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def select_choice(text, choices, policy):
    """Return an offered identity, cancel, or None. Whole-answer matching rejects mixed choices."""
    answer = normalized(text)
    if answer in policy['cancel']:
        return 'cancel'
    matches = []
    for index, identity in enumerate(choices):
        names = policy['names'].get(identity, [identity.replace('_', ' ')])
        names = names + policy['ordinals'][index:index + 1] + [str(index + 1), f'option {index + 1}', normalized(identity)]
        if answer in names:
            matches.append(identity)
    return matches[0] if len(matches) == 1 else None
