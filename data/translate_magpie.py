import logging
import argparse
import os
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
    parser.add_argument("--folder", type=str, default="per_idiom_punct")
    args = vars(parser.parse_args())

    mname = f'Helsinki-NLP/opus-mt-en-nl'
    model = MarianMTModel.from_pretrained(mname)
    tok = MarianTokenizer.from_pretrained(mname)
    if torch.cuda.is_available():
        model = model.cuda()

    batch_size = 16
    TEST_FILE_SRC = args["source"]
    PRED_FILE = args["pred"]

    corpus_length = len(open(TEST_FILE_SRC).readlines())
    logging.info(f"Starting prediction... corpus size: {corpus_length}") 

    prds = dict()
    attention = dict()
    query_attention = dict()
    cross_attention = dict()
    hidden_states_enc = dict()
    hidden_states_dec = dict()

    with open(TEST_FILE_SRC, 'r', encoding="utf-8") as f_in, \
         open(os.path.join(f"{args['folder']}/prds", PRED_FILE), 'w', encoding="utf-8") as f_out:
        lines = f_in.readlines()
        for i in range(0, len(lines)+1, batch_size):
            if not lines[i:i+batch_size]:
                continue
            srcs, annotations = zip(*[x.split("\t")[:2] for x in lines[i:i+batch_size]])
            logging.info(f"Doing {i:4d}/{corpus_length:4d}")
            batch = tok.prepare_seq2seq_batch(
                src_texts=srcs, return_tensors="pt")
            for x in batch:
                if torch.cuda.is_available():
                    batch[x] = batch[x].cuda()

            outputs_beam = model.generate(
                **batch, num_beams=5, max_length=512,
                return_dict_in_generate=True, output_hidden_states=True,
                output_attentions=True)

            outputs_cross_attention = model.forward(
                **batch, return_dict=True, output_hidden_states=True,
                output_attentions=True, decoder_input_ids=outputs_beam["sequences"])

            attention_tmp = torch.stack(
                [x[0] for x in outputs_beam["encoder_attentions"]], dim=0).transpose(0, 1)
            query_attention_tmp = torch.stack(
                [x[1] for x in outputs_beam["encoder_attentions"]], dim=0).transpose(0, 1)
            hidden_states_enc_tmp = torch.stack(
                outputs_beam["encoder_hidden_states"], dim=0).transpose(0, 1)
            cross_attention_tmp = torch.stack(
                outputs_cross_attention["cross_attentions"], dim=0).transpose(0, 1)
            hidden_states_dec_tmp = torch.stack(
                outputs_cross_attention["decoder_hidden_states"], dim=0).transpose(0, 1)

            detokenised_prds = tok.batch_decode(
                outputs_beam["sequences"], skip_special_tokens=True)

            for j, src in enumerate(srcs):
                f_out.write(f"{detokenised_prds[j]}\n")

                src_annotations = []
                for w, l in zip(src.split(), annotations[j].split()):
                    src_annotations.extend([l] * len(tok.tokenize(w)))
                src_len = len(src_annotations) + 1

                prd = tok.convert_ids_to_tokens(outputs_beam["sequences"][j])[1:]
                attention[src] = attention_tmp[j][:, :, :src_len, :src_len].detach().numpy()
                query_attention[src] = query_attention_tmp[j].detach().numpy()
                hidden_states_enc[src] = hidden_states_enc_tmp[j].detach().numpy()
                cross_attention[src] = cross_attention_tmp[j][:, :, :, :src_len].detach().numpy()
                hidden_states_dec[src] = hidden_states_dec_tmp[j].detach().numpy()
                prds[srcs[j]] = prd

    pickle.dump(dict(prds), open(
        f"{args['folder']}/tokenised_prds/{PRED_FILE.replace('.txt', '_prds.pickle')}", 'wb'), protocol=-1)
    pickle.dump(dict(attention), open(
        f"{args['folder']}/attention/{PRED_FILE.replace('.txt', '_attention.pickle')}", 'wb'), protocol=-1)
    pickle.dump(dict(query_attention), open(
        f"{args['folder']}/query_attention/{PRED_FILE.replace('.txt', '_query_attention.pickle')}", 'wb'), protocol=-1)
    pickle.dump(dict(cross_attention), open(
        f"{args['folder']}/cross_attention/{PRED_FILE.replace('.txt', '_cross_attention.pickle')}", 'wb'), protocol=-1)
    pickle.dump(dict(hidden_states_enc), open(
        f"{args['folder']}/hidden_states_enc/{PRED_FILE.replace('.txt', '_hidden_states_enc.pickle')}", 'wb'), protocol=-1)
    pickle.dump(dict(hidden_states_dec), open(
        f"{args['folder']}/hidden_states_dec/{PRED_FILE.replace('.txt', '_hidden_states_dec.pickle')}", 'wb'), protocol=-1)
