import heapq
from wordcloud import WordCloud

import sys

if __name__ == '__main__':

    heap = []

    with open(sys.argv[1]) as input_file:
        for line in input_file:
            print line
            (word, freq) = line.strip().split(' ')
            heap.append( (int(freq), word) )

    word_list = heapq.nlargest(30, heap)

    text = ''
    for item in word_list:
        (freq, word) = item
        print word,
        print freq
        for i in range(int(freq)):
            text += word + ' '
    print text
