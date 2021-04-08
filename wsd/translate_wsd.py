import logging
import argparse
import nltk
import torch
import pickle

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    args = vars(parser.parse_args())

    src = 'en'  # source language
    trg = 'nl'  # target language

    mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'
    model = AutoModelForSeq2SeqLM.from_pretrained(mname)
    tok = AutoTokenizer.from_pretrained(mname)

    TEST_FILE_SRC = args["source"]
    PRED_FILE = args["pred"]

    corpus_length = len(open(TEST_FILE_SRC).readlines())
    logging.info(f"Starting prediction... corpus size: {corpus_length}") 

    prds = dict()
    cross_attention = dict()
    encoder_attention = dict()
    hidden_states = dict()

    with open(TEST_FILE_SRC, 'r', encoding="utf-8") as f_in: #, \
         #open(PRED_FILE, 'w', encoding="utf-8") as f_out:
        lines = f_in.readlines()
        if lines:
            for i in range(0, len(lines)+1, 32):
                if not lines[i:i+32]:
                    continue
                srcs = [x.split("\t")[0] for x in lines[i:i+32]]
                src_lengths = [x.split("\t")[-1] for x in lines[i:i+32]]
                logging.info(f"Doing {i:4d}/{corpus_length:4d}")
                batch = tok.prepare_seq2seq_batch(
                    src_texts=srcs, return_tensors="pt")

                outputs_beam = model.generate(
                    **batch, num_beams=5, return_dict_in_generate=True,
                    output_hidden_states=True, output_attentions=True)

                outputs_cross_attention = model.forward(
                    **batch, return_dict=True,
                    output_attentions=True,
                    decoder_input_ids=outputs_beam["sequences"])

                cross_attention_batch = torch.stack(
                    outputs_cross_attention["cross_attentions"], dim=0).transpose(0, 1)
                encoder_attention_batch = torch.stack(
                    outputs_beam["encoder_attentions"], dim=0).transpose(0, 1)
                hidden_states_tmp_batch = torch.stack(
                    outputs_beam["encoder_hidden_states"], dim=0).transpose(0, 1)

                for j in range(len(srcs)):
                    src_annotations = []
                    for w in srcs[j].split():
                        src_annotations.extend(["_"] * len(tok.tokenize(w)))

                    src_len = len(src_lengths[j].split()) + 1
                    assert src_len == len(src_annotations) + 1

                    prd = tok.convert_ids_to_tokens(outputs_beam["sequences"][j])[1:]
                    cross_attention[srcs[j]] = \
                        cross_attention_batch[j][:, :, :, :src_len].detach().numpy()
                    hidden_states[srcs[j]] = \
                        hidden_states_tmp_batch[j][:, :src_len, :].detach().cpu().numpy()
                    encoder_attention[srcs[j]] = \
                        encoder_attention_batch[j][:, :, :src_len, :src_len].detach().cpu().numpy()
                    prds[srcs[j]] = prd

                logging.info(srcs[j])

    all_encodings = {
        "prds": prds,
        "attention": encoder_attention,
        "cross_attention": cross_attention,
        "hidden_states": hidden_states
    }

    pickle.dump(all_encodings, open(PRED_FILE.replace("txt", "pickle"), 'wb'))
