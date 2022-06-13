import logging
import argparse
import codecs
import torch
from transformers import MarianTokenizer, MarianMTModel


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    args = vars(parser.parse_args())

    mname = f'Helsinki-NLP/opus-mt-en-nl'
    model = MarianMTModel.from_pretrained(mname)
    tok = MarianTokenizer.from_pretrained(mname)
    if torch.cuda.is_available():
        model = model.cuda()

    TEST_FILE_SRC = args["source"]
    PRED_FILE = args["pred"]

    corpus_length = len(open(TEST_FILE_SRC).readlines())
    logging.info(f"Starting prediction... number of sentences in corpus: {corpus_length}") 
    with codecs.open(TEST_FILE_SRC, 'r', encoding="utf-8") as f, \
         codecs.open(PRED_FILE, 'w', encoding="utf-8") as outfile:
        lines = f.readlines()
        for i in range(0, len(lines)+1, 32):
            srcs = lines[i:i+32]
            if not lines[i:i+32]:
                break
            logging.info(f"Doing {i:4d}/{corpus_length:4d}")
            batch = tok.prepare_seq2seq_batch(
                src_texts=[x.split("\t")[0].strip() for x in lines[i:i+32]],
                return_tensors="pt")

            if torch.cuda.is_available():
                for x in batch:
                    batch[x] = batch[x].cuda()

            outputs = model.generate(**batch, num_beams=5, max_length=512,
                                     return_dict_in_generate=True)
            words = tok.batch_decode(outputs["sequences"], skip_special_tokens=True)

            for line in words:
                #line = ' '.join(nltk.word_tokenize(line))
                outfile.write(f"{line}\n")
