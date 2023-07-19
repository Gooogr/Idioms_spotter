import typing as tp

def tokenize_text(text: str, offsets: tp.List[tp.List[int]]=None) -> tp.Tuple[tp.List[str], tp.List[str]]:
    '''
    Apply word tokenization for input text and marked NER tokens. Each token in the source 
    line is assumed to be separated by a space. Code uses IOB format.
    Args:
        text (str): Input text
        offsets (list[list[int]]): comprehended list with start:end indecies of each NER in text
    Returns:
        word_tokens (list[str]): list of word tokens
        pie_tokens (list[str]): list of coresponded NER labels

    Example:
    text: 'The deal was negotiated behind closed doors .'    
    offsets: [[24, 30], [31, 37], [38, 44]]
    
    Example idiom is 'behind closed doors' and function output will be
    ['The', 'deal', 'was', 'negotiated', 'behind', 'closed', 'doors', '.']
    ['O', 'O', 'O', 'O', 'B-PIE', 'I-PIE', 'I-PIE', 'O']
    '''
    word_tokens = []
    pie_tokens = []
    start = 0

    is_first_pie_token = True

    if not offsets:
        word_tokens = text.split()
        pie_tokens = ['O'] * len(word_tokens)
        return word_tokens, pie_tokens

    for offset in offsets:
        offset_start = offset[0]
        offset_end = offset[1]

        # Add tokens before current offset
        substr = text[start:offset_start]
        substr_tokens = substr.split()
        word_tokens.extend(substr_tokens)
        pie_tokens.extend(['O'] * len(substr_tokens))

        # Add offset tokens
        substr = text[offset_start:offset_end]
        substr_tokens = substr.split()
        word_tokens.extend(substr_tokens)

        sbstr_pie_tokens = ['I-PIE'] * len(substr_tokens)
        if is_first_pie_token:
            sbstr_pie_tokens[0] = 'B-PIE'
            is_first_pie_token = False
        pie_tokens.extend(sbstr_pie_tokens)
        start = offset_end

    # Add the substring after the last offset
    substr = text[start:]
    substr_tokens = substr.split()
    word_tokens.extend(substr_tokens)
    pie_tokens.extend(['O'] * len(substr_tokens))

    return word_tokens, pie_tokens


def tokenize_with_context(texts: tp.List[str], offsets: tp.List[tp.List[int]]=None, context_text_number=2):
    """
    Wrapper around `tokenize_text` function to handle all texts. By default MAGPIE and PIE dataset have 5 sentences per object, 
    and only the 3-rd sentece contains PIEs objects. Other texts only for context.
    Args:
        text (list[str]): Input text
        offsets (list[list[int]]): comprehended list with start:end indecies of each NER in text
        context_text_number (int): number of sentence with PIE inside it
    Returns:
        word_tokens (list[str]): list of word tokens
        pie_tokens (list[str]): list of coresponded NER labels
    Example:
    text: ['Hey !', 'Just to clarify .', 'The deal was negotiated behind closed doors .']
    offsets: [[24, 30], [31, 37], [38, 44]]
    context_text_number: 2
    
    Example idiom is 'behind closed doors' and function output will be
    ['Hey', '!', 'Just', 'to', 'clarify', '.', 'The', 'deal', 'was', 'negotiated', 'behind', 'closed', 'doors', '.']
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PIE', 'I-PIE', 'I-PIE', 'O']
    """
    word_tokens, pie_tokens = [], []
    for idx, text in enumerate(texts):
        if idx != context_text_number:
            tmp_word_tokens, tmp_pie_tokens = tokenize_text(text, None)
        else:
            tmp_word_tokens, tmp_pie_tokens = tokenize_text(text, offsets)
        word_tokens.extend(tmp_word_tokens)
        pie_tokens.extend(tmp_pie_tokens)
    return word_tokens, pie_tokens