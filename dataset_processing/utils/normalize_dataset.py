from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_teh_marbuta_ar,
    normalize_alef_maksura_ar,
)
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar


def normalize_arabic_text(text: str) -> str:
    """
    Normalizes a given Arabic text by applying the following steps in order:
    1.  Normalizes alef, ta' marbuta, and ya' variants. [1]
    2.  Removes Arabic diacritics.
    3.  Converts Arabic numerals to Western (English) numerals.

    Args:
        text: The input Arabic string.

    Returns:
        The normalized Arabic string.
    """
    # Step 1: Normalize alef, ta' marbuta, and ya' variants
    text = normalize_alef_ar(text)
    text = normalize_teh_marbuta_ar(text)
    text = normalize_alef_maksura_ar(text)

    # Step 2: Remove all diacritics
    text = dediac_ar(text)

    # Step 3: Convert Arabic numerals to Western numerals
    numeral_map = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for arabic, western in numeral_map.items():
        text = text.replace(arabic, western)

    text = " ".join(simple_word_tokenize(text)) 

    return text


