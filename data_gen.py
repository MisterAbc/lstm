from nltk.corpus import treebank

def make_sentences():
    dictionary = [k.strip() for k in open("./embeddings/words.lst")]
    ind_lookup = {word:(ind+1) for ind,word in enumerate(dictionary)}

    taglst = [k.strip() for k in open("data/tags.lst")]
    tag_lookup = {word:(ind+1) for ind,word in enumerate(taglst)}

    bracket_rep = { "-LRB-":"(",
                    "-RRB-":")",
                    "-RSB-":"[",
                    "-RSB-":"]",
                    "-LCB-":"{",
                    "-RCB-":"}"}

    sentences = list(treebank.tagged_sents())
    for i,sent in enumerate(sentences):
        sent = [(item.lower(),tag) for (item,tag) in sent if tag != '-NONE-']
        sent = [(bracket_rep.get(item, item), tag)                          for (item,tag) in sent]
        sent = [(u'0', tag) if item[0].isdigit() else (item,tag)            for (item,tag) in sent]
        sent = [(u"UNKNOWN", tag) if item not in ind_lookup else (item,tag) for (item,tag) in sent]
        # 1 indexed!!!
        sent = [(ind_lookup[item], tag_lookup[tag])                         for (item,tag) in sent]
        sentences[i] = sent

    sentences = [i for i in sentences if len(i) > 4]
    print(sum(map(len, sentences)) / float(len(sentences)))

    return sentences
            

if __name__ == "__main__":
    sentences = make_sentences()
    sentences = [zip(*k) for k in sentences]
    words,tags = zip(*sentences)
    print(max([len(k) for k in words]))
    open("data/sentences.txt", 'w').write("\n".join([" ".join(str(x) for x in k) for k in words]))
    open("data/tags.txt", 'w').write("\n".join([" ".join(str(x) for x in k) for k in tags]))
