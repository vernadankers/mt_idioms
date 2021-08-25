import logging
import argparse
import nltk
import torch
import pickle

from transformers import MarianTokenizer, MarianMTModel


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--lang", type=str, default="nl")
    parser.add_argument("--batch_size", type=int, default=16)
    args = vars(parser.parse_args())

    src = 'en'  # source language
    trg = args["lang"]  # target language
    batch_size = args["batch_size"]

    mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'
    model = MarianMTModel.from_pretrained(mname)
    tok = MarianTokenizer.from_pretrained(mname)
    if torch.cuda.is_available():
        model = model.cuda()

    TEST_FILE_SRC = args["source"]
    PRED_FILE = args["pred"]

    corpus_length = len(open(TEST_FILE_SRC).readlines())
    logging.info(f"Starting prediction... corpus size: {corpus_length}") 

    prds = dict()
    cross_attention = dict()
    encoder_attention = dict()

    with open(TEST_FILE_SRC, 'r', encoding="utf-8") as f_in:
        lines = f_in.readlines()
        if lines:
            for i in range(0, len(lines)+1, batch_size):
                if not lines[i:i + batch_size]:
                    continue
                srcs = [x.split("\t")[0] for x in lines[i:i + batch_size]]
                src_lengths = [x.split("\t")[-1] for x in lines[i:i + batch_size]]
                logging.info(f"Doing {i:4d}/{corpus_length:4d}")
                batch = tok.prepare_seq2seq_batch(
                    src_texts=srcs, return_tensors="pt")
                if torch.cuda.is_available():
                    for x in batch:
                        batch[x] = batch[x].cuda()

                outputs_beam = model.generate(
                    **batch, num_beams=5, return_dict_in_generate=True, max_length=512,
                    output_hidden_states=True, output_attentions=True)

                outputs_cross_attention = model.forward(
                    **batch, return_dict=True,
                    output_attentions=True,
                    decoder_input_ids=outputs_beam["sequences"])

                cross_attention_batch = torch.stack(
                    [x[0] for x in outputs_cross_attention["cross_attentions"]], dim=0).transpose(0, 1)
                encoder_attention_batch = torch.stack(
                    [x[0] for x in outputs_beam["encoder_attentions"]], dim=0).transpose(0, 1)

                for j in range(len(srcs)):
                    src_len = len(src_lengths[j].split()) + 1

                    prd = tok.convert_ids_to_tokens(outputs_beam["sequences"][j])[1:]
                    cross_attention[srcs[j]] = \
                        cross_attention_batch[j][:, :, :, :src_len].detach().cpu().numpy()
                    encoder_attention[srcs[j]] = \
                        encoder_attention_batch[j][:, :, :src_len, :src_len].detach().cpu().numpy()
                    prds[srcs[j]] = prd

                logging.info(srcs[j])

    all_encodings = {
        "prds": prds,
        "attention": encoder_attention,
        "cross_attention": cross_attention,
    }

    pickle.dump(all_encodings, open(PRED_FILE.replace("txt", "pickle"), 'wb'))
