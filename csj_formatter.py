from typing import List, Tuple
import re

def extract_head_tagged_string(s) -> Tuple[str, str]:
    """文字列の先頭に含まれるタグ付き文字列を抽出する.

    動作例:
        >>> extract_head_tagged_string('(D コレ(F エー)ハイ(? ア,ノ))ハイ(F エー)ハイ')
        ('(D コレ(F エー)ハイ(? ア,ノ))', 'ハイ(F エー)ハイ')

    Args:
        s (str): タグ付き文字列

    Returns:
        Tuple[str, str]: タグ付き文字列とそれ以降の文字列
    """
    if len(s) == 0:
        return '', ''
    if s[0] != '(':
        raise ValueError('not begin with (')
    stack = []
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        elif c == ')':
            stack.pop()
            if len(stack) == 0:
                return s[:i+1], s[i+1:]
    raise ValueError('not end with )')

def split_tagged_content_with_semicolon_or_comma(content: str) -> List[str]:
    """タグが付けられた文字列を，セミコロンかカンマで分割する.
    
    動作例:
        >>> split_tagged_content_with_semicolon_or_comma('ハイ(F ウン)ハイ;ハイ')
        ['ハイ(F ウン)ハイ', 'ハイ']
        >>> split_tagged_content_with_semicolon_or_comma('ハイ(F ウン)ハイ,ハイ')
        ['ハイ(F ウン)ハイ', 'ハイ']

    Args:
        content (str): タグが付けられた文字列

    Returns:
        List[str]: 分割された文字列のリスト
    """
    result = []
    prev = 0
    open_parenthesis_count = 0
    for i, c in enumerate(content):
        if c in (';', ','):
            if open_parenthesis_count == 0:
                result.append(content[prev:i])
                prev = i + 1
        elif c == '(':
            open_parenthesis_count += 1
        elif c == ')':
            open_parenthesis_count -= 1
    result.append(content[prev:])
    return result

def remove_tag_from_plain_tagged_string(s: str):
    """タグ付き文字列からタグを除去する.
    Args:
        s (str): タグ付き文字列

    Returns:
    """

    # 空文字列の場合は～文字列を返す
    if len(s) == 0:
        return ''

    result = ''

    while len(s) > 0:
        # タグが始まる前の部分を取り出す
        match = re.match(r'^([^(]+)(.*)$', s)
        if match:
            result += match.group(1)
            s = match.group(2)
        else:
            # 最長一致でタグの部分を取り出す
            tagged_string, s = extract_head_tagged_string(s)
            match_ = re.match(r'^\(([^ ]+) (.+)\)$', tagged_string)
            if not match_:
                if tagged_string == '(?)':
                    continue
                else:
                    raise Exception('Invalid tagged string: {}'.format(tagged_string))
            tag = match_.group(1)
            content = match_.group(2)
            # タグごとにコンテンツの絞り込みをする
            if tag in ('M', 'O', 'X'): # tag in ('F', 'D', 'D2', 'M', 'O', 'X',):
                pass

            elif tag == 'F':
                # # F の場合はタグ<F></F>をつける
                # content = "<F>" + content + "</F>"
                # F の場合は削除    
                content = ""
            elif tag in ('D', 'D2'):
                # D, D2 の場合は削除
                content = ""

            else:
                # とりあえずセミコロンとカンマで分割
                contents = split_tagged_content_with_semicolon_or_comma(content)
                if tag in ('?', 'K'):
                    # ? の場合は最初の要素だけ
                    content = contents[0]
                elif tag == 'A':
                    if re.match(r'^[０-９．]+$', contents[1]):
                        # content = contents[0]
                        content = contents[1]
                    else:
                        content = contents[1]
                else:
                    raise Exception('Unknown tag: {}'.format(tag))                    
            # content がタグを含む場合は再帰処理をする
            if '(' in content:
                content = remove_tag_from_plain_tagged_string(content)
            result += content
            
    return result
